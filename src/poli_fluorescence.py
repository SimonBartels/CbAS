import os
import warnings

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation
from gfp_gp import SequenceGP
from util import AA_IDX


class FluorescenceFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation("FLUORESCENCE", max_sequence_length=237, aligned=True, alphabet=AA_IDX)

    def create(self, seed=0):
        gp = SequenceGP(load=True, load_prefix=os.path.join(os.path.dirname(__file__), "..", "data", "gfp_gp"))
        assert (gp.X_.shape[1] == self.get_setup_information().get_max_sequence_length())

        class FluorescenceProblem(AbstractBlackBox):
            def _black_box(self, x, context=None):
                # return negative since we want to minimize
                # this assumes a 1-hot encoding
                #return -torch.as_tensor(gp.predict(torch.argmax(x.reshape([x.shape[0], 237, 20]), dim=-1).numpy()))
                return -gp.predict(x)

        # TODO: Why is this a DNA sequence?!
        #raise NotImplementedError("something is seriously wrong!")
        #x0 = torch.as_tensor(one_hot_encode_aa(get_gfp_base_seq()))
        warnings.warn("Wild-type given as DNA sequence!")
        #x0 = torch.as_tensor(one_hot_encode_aa(convert_idx_array_to_aas(gp.X_[:1, :])[0]), dtype=torch.float64).reshape(-1, 1).T
        x0 = gp.X_[:1, :]
        assert(x0.shape[0] == 1 and len(x0.shape) == 2)
        f = FluorescenceProblem(self.get_setup_information().get_max_sequence_length())
        return f, x0, f(x0)


if __name__ == '__main__':
    from poli.core.registry import register_problem
    register_problem(FluorescenceFactory(), "../env")
