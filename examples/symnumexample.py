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
