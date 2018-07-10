# temp fix to pass the unit tests
# will be removed in future releases
# please refer to https://github.com/allenai/allennlp-as-a-library-example/pull/13

import os
from allennlp.common.testing import AllenNlpTestCase
AllenNlpTestCase.MODULE_ROOT = (os.path.dirname(os.path.abspath(__file__)) + '/../')