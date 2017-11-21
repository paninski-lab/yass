from yass import Pipeline, piped_transformation
from yass.filter import butterworth
from yass.standarize import standarize

pipeline = Pipeline(observations='all',
                    input_path='path/to/raw.bin',
                    output_path='path/to/processed.bin',
                    dtype='int16', channels='all',
                    n_channels=512, max_memory='1gb',
                    buffer_size=30, mode='single-channel')

piped_butterworth = piped_transformation(butterworth, low_freq=300,
                                         high_factor=0.1, order=3,
                                         sampling_freq=20000)

piped_standarize = piped_transformation(standarize, srate=2000)


pipeline.add([piped_butterworth, piped_standarize])

pipeline.run()
