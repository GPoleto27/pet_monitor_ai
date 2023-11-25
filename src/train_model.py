from argparse import ArgumentParser
from os import listdir

import cv2
import tensorflow as tf

from model import Model, keras

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', default='models/model.h5')
    parser.add_argument('--dataset', default='data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--plot', default='models/model.png')
    parser.add_argument('--classes', default='models/classes.txt')
    parser.add_argument('--summary', default='models/summary.txt')
    args = parser.parse_args()

    # convert each character in "poleto" to its ASCII code, then concatenate them and convert to int
    seed = int(''.join([str(ord(c)) for c in 'poleto'])) % (2**32 - 1)

    # load dataset
    train_data, validation_data = tf.keras.preprocessing.image_dataset_from_directory(
        args.dataset,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(40, 30),
        batch_size=args.batch_size
    ), tf.keras.preprocessing.image_dataset_from_directory(
        args.dataset,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(40, 30),
        batch_size=args.batch_size
    )

    # create model
    model = Model().model

    # compile model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.build((None, 40, 30, 3))

    model.summary()

    classes = [d for d in listdir(args.dataset) if not d.startswith('.')]

    # save classes to models/classes.txt
    with open(args.classes, 'w') as f:
        f.write('\n'.join(classes))
    # print summary to models/summary.txt
    with open(args.summary, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Train model
    model.fit(train_data, epochs=args.epochs, validation_data=validation_data)
    
    # Test model
    model.evaluate(train_data)
    
    # Save model
    model.save(args.model)
    
    # Test model
    model = keras.models.load_model(args.model)
    model.evaluate(train_data)
    
    # Plot model
    keras.utils.plot_model(model, to_file=args.plot, show_shapes=True, show_layer_names=True)
    


if __name__ == '__main__':
    main()
