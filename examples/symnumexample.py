# +-----------------------------------------------------------------------------+
# |  Copyright 2019-2020 IBM Corp. All Rights Reserved.                         |
# |                                                                             |
# |  Licensed under the Apache License, Version 2.0 (the "License");            |
# |  you may not use this file except in compliance with the License.           |
# |  You may obtain a copy of the License at                                    |
# |                                                                             |
# |      http://www.apache.org/licenses/LICENSE-2.0                             |
# |                                                                             |
# |  Unless required by applicable law or agreed to in writing, software        |
# |  distributed under the License is distributed on an "AS IS" BASIS,          |
# |  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   |
# |  See the License for the specific language governing permissions and        |
# |  limitations under the License.                                             |
# +-----------------------------------------------------------------------------+
# |  Authors: Lorenz K. Mueller, Pascal Stark                                   |
# +-----------------------------------------------------------------------------+

from colna.analyticnetwork import SymNum

amp1 = SymNum(name='a1', default=0.5, product=True)
amp2 = SymNum(name='a2', default=0.8, product=True)

amp3 = amp1 * amp2

print(amp1)
print(amp2)
print(amp3)

# Evaluate without feed dictionary and use individual defaults
print(amp3.eval(feed_dict=None, use_shared_default=False))

# Evaluate without feed dictionary, but use shared defaults
print('amp3 shared default:', amp3.shared_default)
print(amp3.eval(feed_dict=None, use_shared_default=True))

# Evaluate with feed dictionary
feed = {'a1': 2, 'a2': 3}
print(amp3.eval(feed_dict=feed))

# Evaluate with partial feed dictionary
feed = {'a2': 3}
print(amp3.eval(feed_dict=feed))
