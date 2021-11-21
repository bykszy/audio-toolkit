from audiotoolkit import *

#xx2 = AudioClip().cut(10, 0, 0).add_padding(480000).get_mixup(1, 2, 0.3)
#xx = AudioClip().cut(10, 0, 0).add_padding(480000).get_audio()

#xx1_and_fs1 = AudioClip('wav-example.wav').cut(10, 0, 0).add_padding(480000).add_echo(100).reverse().add_noise().erase_freq(240000, 100).get_audio()
#xxspecaug3 = AudioClip('wav-example.wav').cut(10, 0, 0).create_spec().specaug(0, 0, 10, 20)
#xxspecaug = AudioClip('wav-example.wav').cut(10, 0, 0).create_spec().specaug(1, 0, 10, 20)
xxspecaug1 = AudioClip('record.wav.wav').cut(10, 0, 0).create_spec().specaug(1, 1, 10, 20)
xxspecaug2 = AudioClip('toilet_flush.wav.wav').cut(10, 0, 0).create_spec().specaug(1, 1, 10, 20)

#print("xx")
#print(xx)
#print("xxandft")
#print(xx1_and_fs1)

#t = np.arange(0, len(xx[0, 0])) / xx[0, 1]
#fig, axs = plt.subplots(len(xx) + 1)
#fig.suptitle('startowe')
#for i in range(len(xx)):
#    axs[i].plot(t, xx[i, 0])
#axs[len(xx)].plot(t, xx2[0, 0])
#plt.show()
