# first, we need to import our essentia module. It is aptly named 'essentia'!
import sys
sys.path.append('/usr/local/lib/python3/dist-packages/')
import essentia

# we start by instantiating the audio loader:
loader = essentia.standard.MonoLoader(filename='/Users/stefanwijtsma/Downloads/MIDI_XR/Project Files/MIDI XR Demo Project 1/MIDI XR DEMO - Davinchi -Short.mp3')

# and then we actually perform the loading:
audio = loader()

frame = audio[6*44100 : 6*44100 + 1024]
spec = spectrum(w(frame))
mfcc_bands, mfcc_coeffs = mfcc(spec)

plot(spec)
plt.title("The spectrum of a frame:")
show()

plot(mfcc_bands)
plt.title("Mel band spectral energies of a frame:")
show()

plot(mfcc_coeffs)
plt.title("First 13 MFCCs of a frame:")
show()