__author__ = 'Simon Bartels'

import os
import numpy as np
import poli.core.registry
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation
from util import get_experimental_X_y, AA_IDX, build_vae, one_hot_encode_aa


class CBASVAEProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation("CBAS_VAE", max_sequence_length=237, aligned=True, alphabet=AA_IDX)

    def create(self, seed: int = 0):
        X_train, _, _ = get_experimental_X_y(random_state=seed, train_size=5000)
        x0 = X_train[:3, :, :]
        # convert one-hot sequences to index format
        x0 = np.argmax(x0, axis=-1)

        info = self.get_setup_information()
        AA = len(info.get_alphabet())
        L = info.get_max_sequence_length()

        # the objective function does NOT change, it stays the same VAE
        RANDOM_STATE = 1
        train_size_str = "5k"
        vae_suffix = '_%s_%i' % (train_size_str, RANDOM_STATE)

        vae_0 = build_vae(latent_dim=20,
                               n_tokens=20,
                               seq_length=X_train.shape[1],
                               enc1_units=50)

        vae_0.encoder_.load_weights("../models/vae_0_encoder_weights%s.h5" % vae_suffix)
        vae_0.decoder_.load_weights("../models/vae_0_decoder_weights%s.h5" % vae_suffix)
        vae_0.vae_.load_weights("../models/vae_0_vae_weights%s.h5" % vae_suffix)

        vae = vae_0

        class CBASVAEProblem(AbstractBlackBox):
            def _black_box(self, x, context=None):
                # TODO: permute?
                #x1h = tf.one_hot(x, AA)  #.flatten(start_dim=1).double()
                #x_ = tf.flatten(tf.permute(tf.reshape(x1h, [x.shape[0], L, AA]), [0, 2, 1]),
                #                   start_dim=1)
                x1h = np.zeros([1, L, AA])
                x1h[0, np.arange(L), x[0, :]] = 1
                x_ = x1h
                #z = vae.encoder_.predict(x_, batch_size=1)[0]
                y = vae.vae_.evaluate([x_], [x_, np.zeros([1, 20])], batch_size=1, verbose=0)
                #print(y)
                return y[0].reshape(-1, 1)

        f = CBASVAEProblem(info.get_max_sequence_length())
        return f, x0, f(x0)


if __name__ == '__main__':
    poli.core.registry.register_problem(CBASVAEProblemFactory(), "../env")
    from poli.objective_factory import create
    info, f, x, y, _ = create(CBASVAEProblemFactory().get_setup_information().get_problem_name(), observer=None)
    print(f(x))
    problem_factory_name = os.path.basename(__file__)[:-2] + CBASVAEProblemFactory.__name__
