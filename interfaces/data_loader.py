import itertools
import random
import functools
import numpy as np
from utils.parallelism import AsyncPool

# TODO: make sure that multiple data loaders can run, such that each only returns the tags they can generate

INPUT = "input"
OUTPUT = "output"
IDS = "ids"

TRAIN = "training"
TRAINING = "training"
VALIDATE = "validation"
VALIDATION = "validation"
TEST = "test"
TESTING = "test"
UNSUPERVISED = "unsupervised"


class BaseDataLoader(object):
    def __init__(self,
                 sets=None,  # dict with {set_name: percentage}
                 epochs=None,  # number of times to see each sample. None for infinite generator
        ):
        assert not (type(sets) == dict and type(epochs) == int), "Cannot have fixed epoch size with variable set probabilities."

        self.sets = sets
        self.epochs = epochs

    def prepare(self):
        raise NotImplementedError()

    def load_sample(self, sample_id, input_keys_to_do, output_keys_to_do):
        raise NotImplementedError()

    def preprocess_sample(self, chunk_memory, index, sample_data):
        raise NotImplementedError()

    def filter_samples(self):
        raise NotImplementedError()

    @property
    def number_of_samples(self):
        raise NotImplementedError()
    @property
    def number_of_samples_in_iterator(self):
        raise NotImplementedError()

    def generate_batch(self, chunk_size, required_input, required_output):
        raise NotImplementedError()

    def initialize_empty_chunk(self, chunk_size, required_input, required_output):
        raise NotImplementedError()

    def skip_first_chunks(self, n):
        raise NotImplementedError()


class StandardDataLoader(BaseDataLoader):

    OUTPUT_DATA_SIZE_TYPE = {
    }

    # they are shared between all instances of StandardDataLoader to save memory
    indices = dict()
    data = dict()

    def __init__(self,
                 location,
                 preprocessors=list(),
                 multiprocess=True,
                 crash_on_exception=False,
                 process_last_chunk=False,
                 *args, **kwargs):

        super(StandardDataLoader,self).__init__(*args, **kwargs)
        self.preprocessors = preprocessors
        self.location = location
        self.multiprocess = multiprocess
        self.crash_on_exception = crash_on_exception
        self.process_last_chunk = process_last_chunk
        self.skip_chunks = 0

    def _make_sets_dict(self):
        if type(self.sets) == dict:
            return
        elif type(self.sets) == str:
            self.sets = {
                self.sets: 1.0
            }
        elif type(self.sets) == list:
            #TODO: there is a bug here
            self.sets = {
                dataset: 1.0
                for dataset in self.sets
            }
        else:
            raise NotImplementedError("%s is not a support type for sets: %s of type %s" % (str(self.sets), type(self.sets)))

    @property
    def number_of_samples(self):
        self._make_sets_dict()
        return sum([len(self.indices[s]) for s in self.sets])

    @property
    def number_of_samples_in_iterator(self):
        if self.epochs is None:
            return None
        return int(self.epochs * self.number_of_samples)

    def skip_first_chunks(self, n):
        self.skip_chunks += n

    def generate_batch(self,
                       chunk_size,
                       required_input,  # dict with {input_key: input_size}
                       required_output,  # dict with {output_key: output_size}
                       ):

        self._make_sets_dict()
        input_keys_to_do  = required_input.keys()
        output_keys_to_do = required_output.keys()

        def ignore_exceptions_wrapper(gen):
            while True:
                try:
                    yield next(gen)
                except StopIteration:
                    raise
                except Exception as e:
                    print e # or whatever kind of logging you want
                    if self.crash_on_exception:
                        raise

        sample_loader = functools.partial(self.load_sample, input_keys_to_do=input_keys_to_do, output_keys_to_do=output_keys_to_do)
        for i in xrange(len(self.preprocessors)):
            input_keys_to_do += self.preprocessors[i].extra_input_tags_required
            output_keys_to_do += self.preprocessors[i].extra_output_tags_required

            data_loader = ignore_exceptions_wrapper(itertools.imap(sample_loader, itertools.cycle(self.indices[TRAINING])))
            for j in xrange(i-1):
                data_loader = itertools.imap(self.preprocessors[j].process, data_loader)
            self.preprocessors[i].train(data_loader)


        indices_to_sample_from = sum([self.indices[s] for s in self.sets.keys()],[])
        p = np.array(sum([len(self.indices[s])*[1.0*value/len(self.indices[s])] for s, value in self.sets.iteritems()],[]))
        # normalize
        p = p / sum(p)


        if self.epochs is None:
            consume_samples = False
        else:
            consume_samples = True
            if self.epochs==0:
                # in this case, just fill one chunk
                selection = np.random.choice(a=indices_to_sample_from, size=chunk_size, p=p)
                indices_to_sample_from = iter(selection)
            elif isinstance(self.epochs, int):
                indices_to_sample_from = itertools.chain.from_iterable(itertools.repeat(random.sample(indices_to_sample_from, len(indices_to_sample_from)), self.epochs))
            elif isinstance(self.epochs, float):
                selection = np.random.choice(a=indices_to_sample_from, size=int(self.epochs*len(indices_to_sample_from)), p=p)
                indices_to_sample_from = iter(selection)
            else:
                raise Exception("Epochs should be int, float or None")

            indices_to_sample_from = itertools.islice(indices_to_sample_from, self.skip_chunks * chunk_size, None)

        def consume_sample(sample_id, memory_position):
            try:
                sample_data = self.load_sample(sample_id,
                                               input_keys_to_do,
                                               output_keys_to_do)
                if self.remove_this_sample_before_preprocessing(sample_data):
                    return False

                self.preprocess_sample(chunk_memory, memory_position, sample_data)
                if self.remove_this_sample_after_preprocessing(sample_data):
                    return False

                chunk_memory[IDS][memory_position] = sample_id
                return True  # success!
            except Exception as e:
                print e
                import traceback
                traceback.print_exc()
                if self.crash_on_exception:
                    raise
                return False  # failure :-(

        while indices_to_sample_from:
            chunk_memory = self.initialize_empty_chunk(chunk_size, required_input, required_output)

            processes = [False] * chunk_size
            if self.multiprocess:
                pool = AsyncPool(size=chunk_size)

            while sum(processes) != len(processes):
                try:
                    for position in xrange(chunk_size):
                        if processes[position]:  # We already did this one
                            continue

                        if consume_samples:
                            index = indices_to_sample_from.next()
                        else:
                            selection = np.random.randint(0, len(indices_to_sample_from))
                            index = indices_to_sample_from[selection]

                        if self.multiprocess:
                            pool.add_task(position, consume_sample, index, position)
                        else:
                            processes[position] = consume_sample(index, position)

                    if self.multiprocess:
                        processes = pool.get_results()
                        processes = filter(lambda x: x is not None, processes)

                    yield chunk_memory
                except StopIteration:
                    if self.multiprocess:
                        pool.get_results()
                    if self.process_last_chunk:
                        yield chunk_memory
                    return  # there were no more indices to sample from
        return


    def quick_sample_output(self,output_keys_to_do):

        input_keys_to_do = []  # we do not need additional inputs to sample the output
        output_keys_to_do = output_keys_to_do.keys()
        for i in xrange(len(self.preprocessors)):
            input_keys_to_do += self.preprocessors[i].extra_input_tags_required
            output_keys_to_do += self.preprocessors[i].extra_output_tags_required

        sample_data = self.load_sample(0,input_keys_to_do,
                                       output_keys_to_do)
        for preprocessor in self.preprocessors:
            preprocessor.process(sample_data)
        return sample_data[OUTPUT]


    def initialize_empty_chunk(self, chunk_size, required_input, required_output):

        # dict with {input_key: input_size}

        """
        Initialise empty chunk
        """
        no_samples = chunk_size

        result = {
            INPUT: {},
            OUTPUT: {},
            IDS: [None] * chunk_size,
        }

        # we cannot know the output shape after preprocessing, unless we sample it.
        output_sample = self.quick_sample_output(required_output)

        for tag in required_output.keys():
            if tag in output_sample:
                size, dtype = (no_samples, ) + output_sample[tag].shape, output_sample[tag].dtype
                result[OUTPUT][tag] = np.zeros(size, dtype=dtype)

        for tag, size in required_input.iteritems():
            chunk_shape = tuple(i if i is not None else no_samples for i in size)
            result[INPUT][tag] = np.zeros(chunk_shape, dtype="float32")

        return result


    def preprocess_sample(self, chunk_memory, index, sample_data):

        for preprocessor in self.preprocessors:
            preprocessor.process(sample_data)

        for tag in chunk_memory[INPUT]:
            chunk_memory[INPUT][tag][index] = sample_data[INPUT][tag]

        for tag in chunk_memory[OUTPUT]:
            chunk_memory[OUTPUT][tag][index] = sample_data[OUTPUT][tag]

    def remove_this_sample_before_preprocessing(self, sample):
        return False

    def remove_this_sample_after_preprocessing(self, sample):
        return False

def compress_data(original_class):
    import bcolz
    bcolz.cparams(clevel=4, shuffle=1, cname="blosclz")

    orig_prepare = original_class.prepare

    def prepare(self, *args, **kwargs):
        orig_prepare(self, *args, **kwargs)
        for key in self.data.keys():
            self.data[key] = bcolz.carray(self.data[key])

    original_class.prepare = prepare # set the class' __init__ to the new one
    return original_class