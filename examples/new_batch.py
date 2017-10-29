from yass import BatchProcessorFactory
from yass.filter import butterworth
from yass.standarize import standarize

# initialize factory with default settings, note that input_path and dtype
# are missing since we will apply processing to different files
factory = BatchProcessorFactory(observations='all', channels='all',
                                n_channels=512, max_memory='1gb',
                                buffer_size=30, mode='single-channel')

# create a new processor
bp1 = factory.make(input_path='path/to/raw.bin', dtype='int16')

# iterate over batches
for data, data_indexes, buffer_indexes, batch_number, channel_number in bp1:
    # do stuff
    pass

# you can also apply functions
bp2 = factory.make(input_path='path/to/raw.bin', dtype='int16')

# first parameter to apply must be a function whose first parameter is the
# data to be processed, second argument is the path for the output file
# choose 'temp' if you want to save it in a temp location, file will be
# removed when the machine ins restarted, third arugument is a flag to run the
# processing in parallel the rest of the arguments are the arguments for the
# function
path_to_filtered, dtype = bp2.apply(butterworth, output_path='tmp',
                                    parallel=True, low_freq=300,
                                    high_factor=0.1, order=3,
                                    sampling_freq=20000)

# by using the factory, you can easily chain transformations
bp3 = factory.make(path=path_to_filtered, dtype=dtype)

path_to_standarized, dtype = bp3.apply(standarize, output_path='tmp',
                                       parallel=True, srate=20000)

# yass.Pipeline makes chaining transformations even easier
