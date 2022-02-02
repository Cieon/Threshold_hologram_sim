import numpy as bd
from PIL import Image
import struct

class LoadFile:
    def __init__(self, file, bin = False, export = False):
        """
        Loads a file in the Real/Imaginary/Real/... format and export an image of the pattern
        Returns 1D list of the size, Amplitude array and Phase array

        Parameters:
            file: file name or path
            bin: boolean if the values should be binarized at 50% of max value
            export: boolean if the results should be output as an image
        """

        self.bin = bin
        self.export = export

        self.data = open(file, "rb")

    def loadData(self):
        i = 0
        Real, Im = [], []
        while 1:
            try:
                value = struct.unpack('f', self.data.read(4))[
                    0]  # Read the float values from file in format Real/Imaginary/Real/...
                if i % 2 is 0:
                    Real.append(value)
                    # print("R:",value)
                else:
                    Im.append(value)
                    # print("I:", value)
            except:
                print("End of data")
                break
            i += 1
        if len(Real) == len(Im):
            size = int(bd.sqrt(len(Real)))
        else:
            print("Error in data")
            exit(1)

        Amp = [bd.sqrt(Real[x] ** 2 + Im[x] ** 2) for x in range(size ** 2)]  # Amplitude data
        Phase = [bd.arctan2(Im[x], Real[x]) for x in range(size ** 2)]  # Phase data

        print("Amplitude", "\t", "max:", max(Amp), "\t", "min:", min(Amp), "\t", "Data size:", len(Amp))
        print("Phase", "\t", "max:", max(Phase), "\t", "min:", min(Phase), "\t", "Data size:", len(Phase))

        # Rescaling amplitude data into 8 bit range
        rmin = min(Amp)
        Amp = [Amp[x] - rmin for x in range(size ** 2)]
        maximum = max(Amp)
        scale = float(255.0 / maximum)
        Amp = [Amp[x] * scale for x in range(size ** 2)]
        # Rescaling phase data into 8 bit range
        immin = min(Phase)
        Phase = [Phase[x] - immin for x in range(size ** 2)]
        maximum = max(Phase)
        scale = float(255.0 / maximum)
        Phase = [Phase[x] * scale for x in range(size ** 2)]
        if (self.bin is True):
            # Amplitude binarization to 0 and 255
            Amp = bd.array(Amp)
            Amp[Amp >= 128.0] = 255
            Amp[Amp < 128.0] = 0
            # Phase binarization to 0 and 255
            Phase = bd.array(Phase)
            Phase[Phase >= 128.0] = 255
            Phase[Phase < 128.0] = 0
        if(self.export is True):
            # Forming an array from Amplitude data and forming an image
            Amp = bd.reshape(bd.asarray(Amp), (-1, size), order='C')
            Amp = bd.flipud(Amp)
            im = Image.fromarray(Amp)
            im.convert('RGB').save('Amplitude.bmp', format='bmp')
            # Forming an array from Phase data and forming an image
            Phase = bd.reshape(bd.asarray(Phase), (-1, size), order='C')
            Phase = bd.flipud(Phase)
            im2 = Image.fromarray(Phase)
            im2.convert('RGB').save('Phase.bmp', format='bmp')

        return [size, Amp, Phase]

