from colna.analyticnetwork import SymNum

amp1 = SymNum(name='a1', default=3, product=True, global_default=5)
amp2 = SymNum(name='a2', default=4, product=True, global_default=6)

amp3 = amp1 * amp2

print(amp1)
print(amp2)
print(amp3)

# Evaluate without feed dictionary and use individual defaults
print(amp3.eval(feed_dict=None, use_global_default=False))

# Evaluate without feed dictionary, but use global defaults
print('amp3 global default: ', amp3.global_default)
print(amp3.eval(feed_dict=None, use_global_default=True))

# Evaluate with feed dictionary
feed = {'a1': 2, 'a2': 3}
print(amp3.eval(feed_dict=feed))

# Evaluate with partial feed dictionary
feed = {'a2': 3}
print(amp3.eval(feed_dict=feed))
