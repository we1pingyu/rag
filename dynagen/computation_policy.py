from .computation_policy_default import ComputationPolicyImpl
from .computation_policy_streams import ComputationPolicyStream
from .computation_policy_interface import ComputationPolicyInterface
from .computation_policy_alter_stream import ComputationPolicyAlterStream
from .computation_policy_opt import ComputationPolicyOptimize

def get_computation_policy(choice='default'):
    if choice == 'default':
        print("Using default computation policy")
        return ComputationPolicyImpl()
    elif choice == 'stream':
        print("Using stream computation policy")
        return ComputationPolicyStream()
    elif choice == 'alter_stream':
        print("Using alter stream computation policy")
        return ComputationPolicyAlterStream()
    elif choice == 'optimize':
        print("Using optimize computation policy")
        return ComputationPolicyOptimize()

    return ComputationPolicyInterface()