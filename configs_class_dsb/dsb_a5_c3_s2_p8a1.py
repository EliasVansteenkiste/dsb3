import numpy as np
import data_transforms
import data_iterators
import pathfinder
import lasagne as nn
from collections import namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import theano.tensor as T
import utils
import utils_lung

# TODO: import correct config here
candidates_config = 'dsb_c3_s2_p8a1'

restart_from_save = None
rng = np.random.RandomState(42)

predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
candidates_path = predictions_dir + '/%s' % candidates_config
id2candidates_path = utils_lung.get_candidates_paths(candidates_path)

# transformations
p_transform = {'patch_size': (48, 48, 48),
               'mm_patch_size': (48, 48, 48),
               'pixel_spacing': (1., 1., 1.)
               }
n_candidates_per_patient = 4


def data_prep_function(data, patch_centers, pixel_spacing, p_transform,
                       p_transform_augment, **kwargs):
    x = data_transforms.transform_dsb_candidates(data=data,
                                                 patch_centers=patch_centers,
                                                 p_transform=p_transform,
                                                 p_transform_augment=p_transform_augment,
                                                 pixel_spacing=pixel_spacing)
    x = data_transforms.pixelnormHU(x)
    return x


data_prep_function_train = partial(data_prep_function, p_transform_augment=None, p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None, p_transform=p_transform)
data_prep_function_test = partial(data_prep_function, p_transform_augment=None, p_transform=p_transform)

# data iterators
batch_size = 4

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['training'], train_valid_ids['validation']
print 'n train', len(train_pids)
print 'n valid', len(valid_pids)

# TODO get test_pids without hacks
test_pids = ['026470d51482c93efc18b9803159c960','031b7ec4fe96a3b035a8196264a8c8c3','03bd22ed5858039af223c04993e9eb22','06a90409e4fcea3e634748b967993531','07b1defcfae5873ee1f03c90255eb170','0b20184e0cd497028bdd155d9fb42dc9','12db1ea8336eafaf7f9e3eda2b4e4fef','159bc8821a2dc39a1e770cb3559e098d','174c5f7c33ca31443208ef873b9477e5','1753250dab5fc81bab8280df13309733','1cf8e778167d20bf769669b4be96592b','1e62be2c3b6430b78ce31a8f023531ac','1f6333bc3599f683403d6f0884aefe00','1fdbc07019192de4a114e090389c8330','2004b3f761c3f5dffb02204f1247b211','202898fa97c5949fbdc07ae7ff1cd9f0','21b73c938fd7d346ee77a60bd60aaeac','243038f7bb7787497c59bc17f04c6ed9','26142353f46d20c9fdded93f01e2bff4','263a1c3bfa43556623e75ed901e3fd8f','2703df8c469906a06a45c0d7ff501199','2a3e6ecf9499607ef4fd14b436136b0c','2d596b6ead89ab35577fe625a9a17cbb','2eb92d17ca91b393765e8acf069763a6','2f77fd993fbd858dec3c085b9ff1a3a2','3295cec04482210dc6f78c2b4a1d287b','33387bea2cacf6127035cc7033036a02','34037914ceeec5605fc890159dd425c5','38bf066bba822584e14c0af65d4bb5e9','3ee1fd6a0f3f108c3558e6699fb011f2','42b2161e43b4dd0ea94604485976c59c','4434e19303b62ebaecef2596583ff351','4575fe61bf3f536ce6cfeb26fcc2893c','48ab0b98fc7789304c21430978624f32','49433c1588cc078b825a0eff1dc2e816','49c88f7cc77341c9ae4e64243f9912fc','4b28f147cb82baba3edcdbd34ca19085','505405b3e70fb24b92e6a8a5b7ed339c','50cdacec399071cf70d8badd2511d0b3','519ad4ead3e61d2d71088ac8e46f25b6','52f6d741e674f62fbcf73e6ec4f6a472','538543b57d0c8fa0b2b6bb7c84df3f33','5451203688c930484ba1f3c7f1378847','55b06d60e7c0329787f81d1b7cbf9aa0','567547810a1795b9c8e11c15dfd32c34','5791c42d317f34592be9a933c50e68ad','580cffecce8d3d53cde1abb922adf21a','59af702c21840ec18073b6b56c95e7fe','5a42f0a0d1e060531c20d04ed23efc02','5ae9ab473d59cd29262c47a741177b6e','5ce91933688cc8400105bf640ac11535','5d16819bd78c74448ce852a93bf423ad','61017c23bbae6e17062ff582d1a237b3','616f98dab4db03edbad28c73d22468d2','63458b5875a0b223ec21555d17b52fd4','6379e4435f78a5e5c150c32146ece4d4','649fd56ef9809019b57261fcf9574d76','665c1913d8e90e57af3b745349d19537','68f4dff6dd1f135488e83b8a4ee6e20e','6993396b31078993e13cf9c0a6fd470b','6c71617e2cee498fd3dd20956bb90a3b','6d3b16f2e60c3a1a4246f340dba73676','6d3be6081d76d2365b080e599628d3bc','6d43fdb6eb1bec3a5f4febfd442e8c93','6e240f23afa2c1b4352cd0db5d4f357d','6f229187fe608c9eacc567eb74c1458c','7027c0b8c8f8dcc76c6e4ba923d60a2e','70671fa94231eb377e8ac7cba4650dfb','70f4eb8201e3155cc3e399f0ff09c5ef','7191c236cfcfc68cd21143e3a0faac51','763288341ee363a264fe45a28ea28c21','7869cc6bfc3678fec1a81e93b34648cf','7c2fd0d32df5a2780b4b10fdf2f2cdbe','7ce310b8431ace09a91ededcc03f7361','7cf1a65bb0f89323668034244a59e725','7daeb8ef7307849c715f7f6f3e2dd88e','7f096cdfbc2fe03ec7f779278416a78c','7fd5be8ec9c236c314f801384bd89c0c','80938b4f531fa2334c13d829339e1356','80bda1afde73204abd74d1ebd2758382','81bd0c062bfa8e85616878bab90f2314','82b9fb9e238397b2f3bff98975577ff9','83728b6eed98845556bfc870b7567883','84ed26b5d79da321711ed869b3cad2ea','85ab88f093ca53a4fab5654e24c77ebe','85d6fb4a08853d370935a75de7495a27','86ad341b9ac27364f03981f6a775246c','88acee40bb9d8cb06898d1c5de01d3c8','89f003dbfbdbd18a5cdeb9b128cb075b','8a1e5830a16db34b580202f8b6dbbd3d','8b494d14d835dd5ae13dab19b9520a55','8b9a28375988de6ea0b143d48b4a8dc9','8bb7dd5fbfa5ecb95552d9c587f2fea5','8be7a7cc747365030bee8297221ab5bc','8e60f166f1f1dc0d72f997fe1c9e72b4','8e9002a485cbda2b47cd14014d6f1c36','8f517521a2ed576e853fab1907fa5ffd','8fde44df03fb80366c6604db53d3623f','901ed0a38aa16933c04ffd531b0aa2cf','9050cf3aa8371bd7088c4bdf967141d4','9065f2b133129c5747d42db18a424749','931253c408c440a8494dfaa74251efd3','94df6d1ae21c5bfaebe6f8daf8fcd85b','95a98df466d4f6c6689908ea9a8f324b','96042e205dd3dc055f084aaca245e550','96544665531e7f59bc2730e3c5f42e65','96cca9d8e5764daa4bcb6c0ba07735bc','993f1e68290d591f755669e97b49b4f4','995fc0581ed0e3ba0f97dbd7fe63db59','9a378249b799bbcefac2a7de46896c0a','9b871732b3935661e7639e84a6ab9747','9ca18e68b6b8d9c3112b4b69b7d6fad5','9cc74e673ec9807ee055973e1b185624','9de48cf43611478ffc1fef051b75dc8c','a0e60d7a13f6bb4002cc4a08e60b0776','a0fc609febe3eef5a4713a22996cf8e5','a2558184e0f4a68e9fb13579d20cb244','a2a4bc7708f6831470d757cd6f32bffe','a334d15ac8d2d25bce76693b1b2a3ed7','a5bb766ab3b1bc5a8023a50a956595f2','a5d7909f14d43f01f44cdcaabed27b84','a6c15206edadab0270898f03e770d730','aa59b7a4aa4dfb2489feea527eda3e4d','ab9c7bef62d1ad65b824414087b6f06b','ac4056071f3cc98489b9db3aebfe2b6a','ae2fdcd8daa3fede6ae23cc63a8d9a82','ae4e9d8aab8f8f5ae975bcca923f468d','ae61ec94b0b8de5439180f4776551e42','aec5a58fea38b77b964007aa6975c049','af1d0c2fcde369dd1b715460c2f704a2','b0599ad2f33276e7cd065eaa8dcec8a2','b17c07114dcf49ce71c8da4b43cf1192','b4d5b618fdf3a5a1bcfb325a3715e99e','b4db5b96c65a668a2e63f9a3ed36afe7','b53d997901eb880c41fbfbc82847204c','b6857d98b7b3dbe84f153617f4dfd14b','b82efe72526c59a96257208d95e54baf','b8793dbd40de88c0de0913abbaab0fe7','bbf7a3e138f9353414f2d51f0c363561','bdc2daa372a36f6f7c72abdc0b5639d1','bdfb2c23a8c1dca5ea8c1cc3d89efee9','be3e35bf8395366d235b8bcfc71a05ee','be9a2df5a16434e581c6a0625c290591','bf6a7a9ab4e18b18f43129c9e22fb448','c0c5a155e6e59588783c2964975e7e1e','c25876fb40d6f8dafd1ecb243193dd3f','c2ef34cc347bc224b5a123426009d027','c3a9046fbe2b0f0a4e43a669c321e472','c46c3962c10e287f1c1e3af0d309a128','c71d0db2086b7e2024ca9c11bd2ca504','c7bdb83b7ca6269fac16ab7cff930a2e','c87a713d17522698958de55c97654beb','c95f2aa23e6d6702f5b16a3b35f89cf0','cbb9bbd994c235b56fb77429291edf99','cc1b7e34d9eba737c9fb91316463e8f7','cc4805e3ebe8621bc94a621b1714fc84','cd68d1a14cc504e3f7434d5cc324744d','cd6be62834c72756738935f904ec9c2c','cdb53f3be6d8cce07fa41c833488d8a5','d03127f497cae40bcbd9996b4d1f5b90','d032116d73789ff9c805f493357b4037','d1131708024b32032ade1ef48d115915','d1a20ef45bb03f93a407b492066f6d88','d2ec8f0fc56a9168cda0c707e49974ab','d3a8fb1da8f7a0dcbd5a8d65f3647757','d42c998d037fb3003faba541e2cf649a','d4a075768abe7fe43ad1caac92515256','d5a0333be8795805fc39509f817780ee','d654966fd2498de023552b830c07a659','d753676c2c6c8ac6f97bd61ecab7554a','d81852bffda09dc8033a45332397c495','dbd9c8025907511e965e7abad955547d','e0aa61b44c33e6a75940a8541c6894c9','e314fd13809db0132443b924401d828b','e33c25d0dbca5e54385f2100ce523467','e3bc0a970a4af5d52826e06742f90e5b','e42065c1145ccf734312cb9edbe5234b','e60d99ea9648e1ce859eb0b386365e26','e6160ed0ff2eb214abd4df9a3c336c1d','e6d8ae8c3b0817df994a1ce3b37a7efb','e9a27e2645e1fad9434ce765f678585f','ea01deecde93cd9503a049d71d46e6d5','ea3a771ef05e288409e0250ea893cf87','eaeebb7a63edc8a329a7c5fbc583a507','eb9db3f740f8e153e85f83c57bc4e522','ebcdfabecf4b46b1e55e4a4c75a0afb0','efcb6def7a2080243052b6046186ab24','f0310ffc724faf9f7aef2c418127ee68','f4d23e0272a2ce5bfc7f07033d4f2e7d','f5ff7734997820b45dafa75dff60ece8','f7c387290d7e3074501eac167c849000','f89e3d0867e27be8e19d7ed50e1eb7e8','fad57a1078ddbc685e517bd8f24aa8ac','fb55849cee6473974612c17f094a38cd','fb5874408966d7c6bebd3d84a5599e20','fcfab3eddbdf0421c39f71d651cc5c56','fdcd385b0d2d12341661e1abe845be0b','ff8599dd7c1139be3bad5a0351ab749a']

train_data_iterator = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=batch_size,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_train,
                                                              id2candidates_path=id2candidates_path,
                                                              rng=rng,
                                                              patient_ids=train_pids,
                                                              random=True, infinite=True)

valid_data_iterator = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=1,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_valid,
                                                              id2candidates_path=id2candidates_path,
                                                              rng=rng,
                                                              patient_ids=valid_pids,
                                                              random=False, infinite=False)

test_data_iterator = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                             batch_size=1,
                                                             transform_params=p_transform,
                                                             n_candidates_per_patient=n_candidates_per_patient,
                                                             data_prep_fun=data_prep_function_test,
                                                             id2candidates_path=id2candidates_path,
                                                             rng=rng,
                                                             patient_ids=test_pids,
                                                             random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / batch_size
max_nchunks = nchunks_per_epoch * 10

validate_every = int(0.5 * nchunks_per_epoch)
save_every = int(0.25 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-5,
    int(5 * nchunks_per_epoch): 2e-6,
    int(6 * nchunks_per_epoch): 1e-6,
    int(7 * nchunks_per_epoch): 5e-7,
    int(9 * nchunks_per_epoch): 2e-7
}

# model
conv3 = partial(dnn.Conv3DDNNLayer,
                pad="valid",
                filter_size=3,
                nonlinearity=nn.nonlinearities.rectify,
                b=nn.init.Constant(0.1),
                W=nn.init.Orthogonal("relu"))

max_pool = partial(dnn.MaxPool3DDNNLayer,
                   pool_size=2)


def dense_prelu_layer(l_in, num_units):
    l = nn.layers.DenseLayer(l_in, num_units=num_units, W=nn.init.Orthogonal(),
                             nonlinearity=nn.nonlinearities.linear)
    l = nn.layers.ParametricRectifierLayer(l)
    return l


def build_model():
    l_in = nn.layers.InputLayer((None, n_candidates_per_patient, 1,) + p_transform['patch_size'])
    l_in_rshp = nn.layers.ReshapeLayer(l_in, (-1, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((batch_size,))

    l = conv3(l_in_rshp, num_filters=128)
    l = conv3(l, num_filters=128)

    l = max_pool(l)

    l = conv3(l, num_filters=128)
    l = conv3(l, num_filters=128)

    l = max_pool(l)

    l = conv3(l, num_filters=256)
    l = conv3(l, num_filters=256)
    l = conv3(l, num_filters=256)

    num_units_dense = 512
    l_d01 = dense_prelu_layer(l, num_units=512)
    l_d01 = nn.layers.ReshapeLayer(l_d01, (-1, n_candidates_per_patient, num_units_dense))
    l_d02 = dense_prelu_layer(l_d01, num_units=512)
    l_out = nn.layers.DenseLayer(l_d02, num_units=2,
                                 W=nn.init.Constant(0.),
                                 b=np.array([np.log((1397. - 362) / 1398), np.log(362. / 1397)], dtype='float32'),
                                 nonlinearity=nn.nonlinearities.softmax)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = T.cast(T.flatten(nn.layers.get_output(model.l_target)), 'int32')
    p = predictions[T.arange(predictions.shape[0]), targets]
    p = T.clip(p, epsilon, 1.)
    loss = T.mean(T.log(p))
    return -loss


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates
