# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with pt_rami.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Algorithms
## TF
from spinup.algos.tf1.ddpg.ddpg import ddpg as ddpg_tf1
from spinup.algos.tf1.ppo.ppo import ppo as ppo_tf1
from spinup.algos.tf1.sac.sac import sac as sac_tf1
from spinup.algos.tf1.td3.td3 import td3 as td3_tf1
from spinup.algos.tf1.trpo.trpo import trpo as trpo_tf1
from spinup.algos.tf1.vpg.vpg import vpg as vpg_tf1

from spinup.algos.tf1.memb.memb import memb as memb_tf1


## PT
from spinup.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from spinup.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from spinup.algos.pytorch.sac.sac import sac as sac_pytorch
from spinup.algos.pytorch.td3.td3 import td3 as td3_pytorch
from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from spinup.algos.pytorch.vpg.vpg import vpg as vpg_pytorch

from spinup.algos.pytorch.memb.memb import memb as memb_pytorch



# from spinup.algos.pt_rami.memb.memb import sac as memb_pt

# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__
