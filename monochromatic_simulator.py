import numpy

from diffractsim import colour_functions as cf
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from pathlib import Path
import PIL.ImageOps
from PIL import Image
import scipy.integrate as integrate

import numpy as np

import struct

m = 1.
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9


class MonochromaticField:
    def __init__(self, wavelength, extent_x, extent_y, Nx, Ny, intensity=0.1, export=False):
        """
        Initializes the field, representing the cross-section profile of a plane wave

        Parameters:
            wavelength: wavelength of the plane wave
            extent_x: length of the rectangular grid
            extent_y: height of the rectangular grid
            Nx: horizontal dimension of the grid
            Ny: vertical dimension of the grid
            intensity: intensity of the field
            export: path for field output as an image whenever the plot method is used
        """
        global bd
        bd = numpy

        self.size = 1

        self.I0 = 0  # Variable holding the intensity saved using the integrate_intensity method
        self.w0 = 0  # Variable holding the value of the defined beam width
        self.thr = 0  # Variable holding the value of the defined cell threshold
        self.modulation = 1  # Variable holding the value of phase-amplitude modulation mode (0-phase, 1-amplitude)
        self.steps = 0  # Variable holding the amount of extra superpixels added into the elementary cell
        self.extent_x = extent_x  # Variable holding the extent of the defined field in x axis
        self.extent_y = extent_y  # Variable holding the extent of the defined field in y axis

        self.Nx = bd.int(Nx)  # Making sure Nx and Ny are integrals
        self.Ny = bd.int(Ny)

        self.x = bd.linspace(-extent_x / 2, extent_x / 2,
                             self.Nx)  # Linear space over the x extent containing Nx values
        self.y = bd.linspace(-extent_y / 2, extent_y / 2,
                             self.Ny)  # Linear space over the y extent containing Ny values
        self.xx, self.yy = bd.meshgrid(self.x,
                                       self.y)  # Grids defining to the position of a point in the grid within the x or y axis

        self.E = bd.ones((int(self.Ny), int(self.Nx))) * bd.sqrt(
            intensity)  # Base value of the E field as sqrt of set intensity
        self.λ = wavelength  # Value holding the defined wavelength
        self.intensity = intensity  # Value holding the defined intensity
        self.z = 0  # Value holding the distance of propagation during simulation
        self.cs = cf.ColourSystem(clip_method=0)  # Colour space used

    def add_rectangular_slit(self, x0, y0, width, height):
        """
        Creates a slit centered at the point (x0, y0) with width width and height height
        """
        t = bd.select(
            [
                ((self.xx > (x0 - width / 2)) & (self.xx < (x0 + width / 2)))
                & ((self.yy > (y0 - height / 2)) & (self.yy < (y0 + height / 2))),
                True,
            ],
            [bd.ones(self.E.shape), bd.zeros(self.E.shape)],
        )
        self.E = self.E * t

        self.I = bd.real(self.E * bd.conjugate(self.E))

    def add_circular_slit(self, x0, y0, R):
        """
        Creates a circular slit centered at the point (x0,y0) with radius R
        """

        t = bd.select(
            [(self.xx - x0) ** 2 + (self.yy - y0) ** 2 < R ** 2, bd.full(self.E.shape, True, dtype=bool)],
            [bd.ones(self.E.shape), bd.zeros(self.E.shape)]
        )

        self.E = self.E * t
        self.I = bd.real(self.E * bd.conjugate(self.E))

    def add_gaussian_beam(self, w0, x0=0, y0=0):
        """
        Creates a Gaussian beam with radius equal to w0, centered at x0, y0
        """

        r2 = (self.xx - x0) ** 2 + (self.yy + y0) ** 2
        self.E = self.E * bd.exp(-r2 / (w0 ** 2))
        self.I = bd.real(self.E * bd.conjugate(self.E))

    def add_diffraction_grid(self, D, a, Nx, Ny):
        """
        Creates a diffraction_grid with Nx *  Ny slits with separation distance D and width a
        """

        E0 = bd.copy(self.E)
        t = 0

        b = D - a
        width, height = Nx * a + (Nx - 1) * b, Ny * a + (Ny - 1) * b
        x0, y0 = -width / 2, height / 2

        x0 = -width / 2 + a / 2
        for _ in range(Nx):
            y0 = height / 2 - a / 2
            for _ in range(Ny):
                t += bd.select(
                    [
                        ((self.xx > (x0 - a / 2)) & (self.xx < (x0 + a / 2)))
                        & ((self.yy > (y0 - a / 2)) & (self.yy < (y0 + a / 2))),
                        True,
                    ],
                    [bd.ones(self.E.shape), bd.zeros(self.E.shape)],
                )
                y0 -= D
            x0 += D
        self.E = self.E * t
        self.I = bd.real(self.E * bd.conjugate(self.E))

    def add_aperture_from_image(self, path, pad=None, Nx=None, Ny=None, invert=False):
        """
        Load the image specified at "path" as a numpy graymap array.
        - If Nx and Ny is specified, we interpolate the pattern with interp2d method to the new specified resolution.
        - If pad is specified, we add zeros (black color) padded to the edges of each axis.
        """

        img = Image.open(Path(path))
        if invert is True:
            img = PIL.ImageOps.invert(img)
        img = img.convert("RGB")
        imgRGB = np.asarray(img) / 256.0
        imgR = imgRGB[:, :, 0]
        imgG = imgRGB[:, :, 1]
        imgB = imgRGB[:, :, 2]
        t = 0.2989 * imgR + 0.5870 * imgG + 0.1140 * imgB

        fun = interp2d(
            np.linspace(0, 1, t.shape[1]),
            np.linspace(0, 1, t.shape[0]),
            t,
            kind="cubic",
        )
        t = fun(np.linspace(0, 1, self.Nx), np.linspace(0, 1, self.Ny))

        # optional: add zeros and interpolate to the new specified resolution
        if pad != None:

            if bd != np:
                self.E = self.E.get()

            Nxpad = int(np.round(self.Nx / self.extent_x * pad[0]))
            Nypad = int(np.round(self.Ny / self.extent_y * pad[1]))
            self.E = np.pad(self.E, ((Nypad, Nypad), (Nxpad, Nxpad)), "constant")
            t = np.pad(t, ((Nypad, Nypad), (Nxpad, Nxpad)), "constant")
            self.E = np.array(self.E * t)

            scale_ratio = self.E.shape[1] / self.E.shape[0]
            self.Nx = int(np.round(self.E.shape[0] * scale_ratio)) if Nx is None else Nx
            self.Ny = self.E.shape[0] if Ny is None else Ny
            self.extent_x += 2 * pad[0]
            self.extent_y += 2 * pad[1]

            fun = interp2d(
                np.linspace(0, 1, self.E.shape[1]),
                np.linspace(0, 1, self.E.shape[0]),
                self.E,
                kind="cubic",
            )
            self.E = bd.array(fun(np.linspace(0, 1, self.Nx), np.linspace(0, 1, self.Ny)))

            # new grid units
            self.x = bd.linspace(-self.extent_x / 2, self.extent_x / 2, self.Nx)
            self.y = bd.linspace(-self.extent_y / 2, self.extent_y / 2, self.Ny)
            self.xx, self.yy = bd.meshgrid(self.x, self.y)

        else:
            self.E = self.E * bd.array(t)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))

    def add_lens(self, f):
        """add a thin lens with a focal length equal to f """
        self.E = self.E * bd.exp(-1j * bd.pi / (self.λ * f) * (self.xx ** 2 + self.yy ** 2))

    def propagate(self, z):
        """Compute the field in distance equal to z with the angular spectrum method
        Parameters:
            z: distance of propagation
        """

        self.z += z

        # Compute angular spectrum
        fft_c = bd.fft.fft2(self.E)
        c = bd.fft.fftshift(fft_c)

        # Calculating the plane wave directions
        kx = bd.linspace(
            -bd.pi * self.Nx // 2 / (self.extent_x / 2),
            bd.pi * self.Nx // 2 / (self.extent_x / 2),
            self.Nx,
        )
        ky = bd.linspace(
            -bd.pi * self.Ny // 2 / (self.extent_y / 2),
            bd.pi * self.Ny // 2 / (self.extent_y / 2),
            self.Ny,
        )
        kx, ky = bd.meshgrid(kx, ky)
        kz = bd.sqrt((2 * bd.pi / self.λ) ** 2 - kx ** 2 - ky ** 2)

        # Propagate the angular spectrum by distance z
        E = bd.fft.ifft2(bd.fft.ifftshift(c * bd.exp(1j * kz * z)))
        self.E = E

        # Compute field Intensity
        self.I = bd.real(E * bd.conjugate(E))

    def get_colors(self):
        """Compute RGB colors"""

        rgb = self.cs.wavelength_to_sRGB(self.λ / nm, 10 * self.I.flatten()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb

    def compute_colors_at(self, z):
        """Propagate the field to a distance equal to z and compute the RGB colors of the beam profile

        Parameters:
            z: distance of the calculation
        """
        self.propagate(z)
        rgb = self.get_colors()
        return rgb

    def plot(self, rgb, figsize=(6, 6), xlim=None, ylim=None, export=None):
        """Visualize the diffraction pattern with matplotlib

        Parameters:
            rgb: return of the get_colors function acquired with either the visualize() or compute_colors_at(z) function
            figsize: (x,y) size of the plot
            xlim: (-x, x) limits of the plot
            ylim: (-y, y) limits of the plot
            export: export path for file with filename and extension, else export as ./last_sim.png
        """

        # plt.style.use("dark_background")
        if bd != np:
            rgb = rgb.get()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        if xlim != None:
            ax.set_xlim(xlim)

        if ylim != None:
            ax.set_ylim(ylim)

        # we use mm by default
        ax.set_xlabel("[mm]")
        ax.set_ylabel("[mm]")

        ax.set_title("Screen distance = " + str(self.z * 100) + " cm")

        im = ax.imshow(
            (rgb),
            extent=[
                -self.extent_x / 2 / mm,
                self.extent_x / 2 / mm,
                -self.extent_y / 2 / mm,
                self.extent_y / 2 / mm,
            ],
            interpolation="spline36",
        )
        if export is not None:
            plt.savefig(export)
        else:
            plt.savefig('./last_sim.png')
        plt.show()

    def add_spatial_noise(self, noise_radius, f_mean, f_size, N=30, A=1):
        """
        Add spatial noise following a radial normal distribution

        Parameters:
            noise_radius: maximum radius affected by the spatial noise
            f_mean: mean spatial frequency of the spatial noise
            f_size: spread spatial frequency of the noise
            N: number of samples
            A: amplitude of the noise
        """

        def random_noise(xx, yy, f_mean, A):
            A = bd.random.rand(1) * A
            phase = bd.random.rand(1) * 2 * bd.pi
            fangle = bd.random.rand(1) * 2 * bd.pi
            f = bd.random.normal(f_mean, f_size / 2)

            fx = f * bd.cos(fangle)
            fy = f * bd.sin(fangle)
            return A * bd.exp((xx ** 2 + yy ** 2) / (noise_radius * 2) ** 2) * bd.sin(
                2 * bd.pi * fx * xx + 2 * bd.pi * fy * yy + phase)

        E_noise = 0
        for i in range(0, N):
            E_noise += random_noise(self.xx, self.yy, f_mean, A) / bd.sqrt(N)

        self.E += E_noise * bd.exp(-(self.xx ** 2 + self.yy ** 2) / (noise_radius) ** 2)
        self.I = bd.real(self.E * bd.conjugate(self.E))

    def visualize(self):
        """
        Visualizes the optical field at the current distance
        """
        return self.compute_colors_at(0)

    def phase_sampling(self, Phase, w0, size, sampling=16, threshold=0, amp_mod=1, steps=0, extended=False,
                       verbose=False):
        """
            Supersamples the phase with gaussian beams

            Parameters:
                Phase: the phase data from LoadFile(path).loadData() function
                w0: writing gaussian beam width (use x * mm/um/nm)
                size: data size used
                sampling: number of pixels supersampling each phase pixel (one axis)
                threshold: threshold of the beam writing the cell
                amp_mod: parameter of amplitude or phase modulation in the cell (0-phase, 1-amplitude) with linear change inbetween
                steps: amount of additional steps between 0 and 1 values in amplitude modulation (amp_mod =1)
                extended: amount of pixel overlap between cells in the overall pattern
                verbose: boolean parameter responsible for showing the graphs of the elementary cell and the full field
        """

        self.w0 = w0
        self.thr = threshold
        self.modulation = amp_mod
        self.steps = steps

        if extended < 0 or (extended > 0 and amp_mod < 1) or (steps > 0 and amp_mod < 1) or bd.abs(amp_mod) > 1:
            print("Function argument error!")
            exit()

        """
        # Create a cell of the supersampling
        cell = bd.ones((sampling, sampling), dtype=complex)
        side = self.extent_x / size  # Size of one cell

        # Create a x,y space for the cell
        cell_x = bd.linspace(-side / 2, side / 2, sampling)
        cell_y = bd.linspace(-side / 2, side / 2, sampling)
        cell_xx, cell_yy = bd.meshgrid(cell_x, cell_y)
        """

        cell = bd.ones((sampling + 2 * extended, sampling + 2 * extended), dtype=complex)
        side = self.extent_x / size  # Size of one cell

        # Create a x,y space for the cell
        ext_mult = (sampling + extended) / sampling
        cell_x = bd.linspace(-side * ext_mult / 2, side * ext_mult / 2, sampling + 2 * extended)
        cell_y = bd.linspace(-side * ext_mult / 2, side * ext_mult / 2, sampling + 2 * extended)
        cell_xx, cell_yy = bd.meshgrid(cell_x, cell_y)

        # Write a gaussian beam with w0 radius
        r2 = (cell_xx) ** 2 + (cell_yy) ** 2
        cell = cell * bd.exp(-r2 / (w0 ** 2))
        # Limit signal by threshold
        cell_gauss = bd.copy(cell)
        cell[cell_gauss < threshold] = (1 - amp_mod) * bd.exp(complex(0, (bd.pi - amp_mod * bd.pi)))
        cell[cell_gauss >= threshold] = (1) * bd.exp(0j)

        # Adding dampening steps between 1 and 0 values (superpixels)
        if int(amp_mod) is 1:
            for s in range(steps):
                # Value of the current step following the curve 1/{1+e^[-5(1-(s+1)/(steps+1)-0.6)]}
                step_value = 1 / (1 + bd.exp(-5 * (0.4 - (s + 1) / (steps + 1))))
                cell_copy = bd.copy(cell)
                for i in range(sampling + 2 * extended):
                    for j in range(sampling + 2 * extended):
                        if cell[i][j] == 0.0:
                            sum = 0
                            # Overwriting the neighbours of current outer values in a cell copy
                            if i > 0:
                                sum += cell[i - 1][j]
                            if i < (15 + 2 * extended):
                                sum += cell[i + 1][j]
                            if j > 0:
                                sum += cell[i][j - 1]
                            if j < (15 + 2 * extended):
                                sum += cell[i][j + 1]
                            if sum > 0:
                                cell_copy[i][j] = step_value
                cell = cell_copy
            cell_I = bd.real(cell * bd.conjugate(cell))

        # Cell plot
        if verbose is True:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(12, 6)
            fig.suptitle("Elementary writing cell")
            im1 = ax1.imshow(bd.absolute(cell), cmap='gray')
            ax1.set_title("Amplitude")
            fig.colorbar(im1, ax=ax1)
            im2 = ax2.imshow(bd.angle(cell), cmap='gray')
            ax2.set_title("Phase")
            fig.colorbar(im2, ax=ax2)
            plt.show()

        # Phase binarization to 0 and 1
        Phase[Phase > 0] = 1

        # Zeroing the current field
        # self.E = bd.zeros(self.E.shape)
        self.E = bd.full(self.E.shape, self.intensity * (1 - amp_mod) * bd.exp(complex(0, (bd.pi - amp_mod * bd.pi))))

        # Writing cells for High values in the Phase pattern, checking for extended status
        for y in range(size):
            for x in range(size):
                if Phase[y][x] == 1:
                    for i in range(sampling + 2 * extended):
                        for j in range(sampling + 2 * extended):
                            if int(amp_mod) is 1 and extended > 0:  # Handling of the extended method of writing
                                if (y in [0, size - 1]) or (x in [0, size - 1]):
                                    # Boundary cases written as regular
                                    if (i >= sampling + extended or j >= sampling + extended):
                                        continue
                                    self.E[y * sampling + i - extended][x * sampling + j - extended] += self.intensity * cell[i + extended][j + extended]
                                else:
                                    # Non boundary cells
                                    self.E[y * sampling + i - extended][x * sampling + j - extended] += self.intensity * cell[i][j]
                            else:  # Regular overwrite method of the necessary hologram points
                                self.E[y * sampling + i - extended][x * sampling + j - extended] = self.intensity * cell[i][j]
        self.I = bd.real(self.E * bd.conjugate(self.E))

        # Full pattern plot
        if verbose is True:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(12, 6)
            fig.suptitle("Full pattern plot")
            im1 = ax1.imshow(bd.absolute(self.E), cmap='gray')
            ax1.set_title("Amplitude")
            fig.colorbar(im1, ax=ax1)
            im2 = ax2.imshow(bd.angle(self.E), cmap='gray')
            ax2.set_title("Phase")
            fig.colorbar(im2, ax=ax2)
            plt.show()

    def export_E(self):
        """Exports the current field as a file in Real/Imaginary/Real/... format named 'ReIm.tab'
        """
        values = []
        for y in range(0, self.Ny):
            for x in range(0, self.Nx):
                values.append(np.real(self.E[y][x]))
                values.append(np.imag(self.E[y][x]))
        from array import array
        fileout = open("ReIm.tab", "wb")
        float_array = array('f', values)
        float_array.tofile(fileout)
        fileout.close()
        print("Exported field to a file")

    def spherical_wave(self, r0=(0 * um, 0 * um), z0=1000 * um):
        """Simulates a spherical wave propagating from a point at (x0, y0, z0)
         Parameters:
            r0 (float, float): (x,y) position of source
            z0 (float): z position of source
        """
        k = 2 * bd.pi / self.λ
        x0, y0 = r0

        R2 = (self.xx - x0) ** 2 + (self.yy - y0) ** 2
        Rz = bd.sqrt((self.xx - x0) ** 2 + (self.yy - y0) ** 2 + z0 ** 2)
        # Modifying the field
        self.E = self.E * bd.exp(-1.j * bd.sign(z0) * k * Rz) / Rz

    def spatial_frequency(self, size):
        """
        Returns the spatial frequency of the pattern
        """
        angle = bd.arcsin(self.λ / (self.extent_x / size))
        return bd.tan(angle)

    def integrate_intensity(self):
        self.I0 = bd.sum(self.I)

    def data_analysis(self, size, order=1, export=False):
        """
        Calculates the diffraction efficiency of the n-th order, contrast and speckle noise of the image
        Parameters:
            size: data size used
            order: order of image to analyze
            export: boolean of data output in array [diff_eff, contrast, speckle]
        """
        # Using the spatial frequency to get the pixel size of one image
        freq = self.spatial_frequency(size)
        freq_pixel = int(freq / self.extent_x * self.Nx)
        # Side of the analysis square (90% of the spatial frequency)
        side = int(0.9 * freq_pixel * self.z)

        I1 = 0  # Value adding up the current pattern intensity
        # 2 1D arrays
        I_a = bd.zeros(side ** 2)  # Array copying the image
        I_t = bd.zeros(side ** 2)  # Array for testing whether a pixel is the signal or background

        a = 0  # Iterator
        # Iterating over the image
        for y in range(int(self.Ny / 2) + int(freq_pixel * self.z * (abs(order) - 1 + 0.05)),
                       int(self.Ny / 2) + int(freq_pixel * self.z * (abs(order) - 1 + 0.05)) + side):
            for x in range(int(self.Nx / 2) + int(freq_pixel * self.z * (abs(order) - 1 + 0.05)),
                           int(self.Nx / 2) + int(freq_pixel * self.z * (abs(order) - 1 + 0.05)) + side):
                I1 += self.I[y][x]  # Integrating the pattern intensity
                I_a[a] += self.I[y][x]  # Copying the image into a 1D array
                # Averaging the intensity of a pixel using weight and 4 neighbours and adding it into a 1D array
                I_t[a] += (2 * self.I[y][x] + (
                            self.I[y + 1][x] + self.I[y - 1][x] + self.I[y][x + 1] + self.I[y][x - 1])) / 6
                a += 1  # Updating the iterator

        # if I0 > 0, else inform to integrate intensity

        # Fraction of intensity in first order of diffraction
        int_1st = I1 / self.I0

        # Separate signal from background based on the simulation parameters (using the neighbour weight averaged intensity)
        if self.steps > 0:
            step_count = self.steps
        else:
            step_count = 1
        I_t[I_t < self.intensity
            * int_1st
            / (self.z ** 2)
            * ((self.extent_x / 30 / mm) ** 3)
            * ((self.w0 / (0.08 * mm * self.extent_x / (30 * mm))) ** 2)
            / ((bd.sqrt(self.thr / 0.5)) ** (1 + self.modulation))
            * (bd.sqrt(step_count))
            * (1 + 5 * (1 - self.modulation) ** 2)] = 0

        # Clipping bright spots way higher than signal average intensity
        I_t[I_t > (bd.mean(I_t) * 25)] = 0

        # Background
        i = bd.where(I_t == 0)
        # Signal
        j = bd.where(I_t > 0)

        # Signal as the non-zero values in the I_t array, set within the I_a array
        # Background as the zero values in the I_t array, set within the I_a array
        signal = bd.copy(I_a)
        background = bd.copy(I_a)
        signal[i] = 0
        signal = signal[signal > 0]
        background[j] = 0
        background = background[background > 0]

        """
        # Histogram of pixel intensity of the image
        plt.hist(signal, bins='auto')
        plt.show()
        """

        # Statistics calculation of the signal and background
        deviation = bd.std(signal)
        signal_sum = bd.sum(signal)
        signal = bd.mean(signal)
        background = bd.mean(background)

        diff_eff = signal_sum / self.I0
        contrast = signal / background
        speckle = deviation / signal

        print("I0=", self.I0, "\tI=", signal_sum, "\tDiffraction efficiency=", "{:.1%}".format(diff_eff))
        print("Average intensity for signal:", signal, "\tbackground:", background, "\tcontrast:", contrast)
        print("Standard deviation for signal: ", deviation, "\tSpeckle ratio:", "{:.1%}".format(speckle))

        # Plotting of the signal
        I_a[i] = 0
        I_a = bd.reshape(I_a, (-1, side), order='C')
        plt.imshow(I_a, cmap='gray')
        plt.title("Signal in the first order")
        plt.colorbar()
        plt.show()

        # Table export of data
        if export is True:
            data = [diff_eff, contrast, speckle]
            return data
