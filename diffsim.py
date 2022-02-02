import diffractsim
import menu as mn
from loadfile import LoadFile
from diffractsim import mm, nm, cm
from monochromatic_simulator import MonochromaticField
from datetime import datetime
import os
import sys

# Argument handling
arguments = sys.argv

menu = 1
if len(arguments) > 1:
    if arguments[1] is '0':
        print('Manual mode selected, set parameters in script')
        menu = 0
    else:
        menu = 1

# diffractsim.set_backend("CPU")

file_path = 'eight_128px_complex.TAB'  # Default file path, variable holding the path
sim_names = ['ssampling', 'wavelength [nm]', 'extent_x [mm]', 'extent_y [mm]', 'intensity', 'distance [cm]', 'w0 [um]',
             'threshold', 'amp_mod', 'steps', 'extended', 'verbose']  # Names of parameters of simulation
sim_var = [16, 632.8, 30, 30, 0.1, 100, 80, 0.5, 1.0, 0, 0,
           0]  # Default values of simulation parameters in order of sim_names
loop_names = ['enabled', 'looping parameter', 'starting value', 'final value', 'step']  # Names of loop parameters
loop_var = [0, 1, 10, 100, 10]  # Default values of loop parameters in order of loop_names
loop_mapping = [0, 1, 5, 6, 7, 8, 9, 10]  # Mapping looping parameter selection to correct simulation variable

if menu is 1:
    mn.print_welcome()  # Menu welcome
    file_path, sim_var, loop_var = mn.run_menu(file_path, sim_names, sim_var, loop_names, loop_var)  # Menu process

loaded_file = LoadFile(file_path, True, True)  # Loading the file from the file_path
fileData = loaded_file.loadData()  # Processing the file
size = fileData[0]  # Size of the hologram in one axis
Amp = fileData[1]  # Amplitude data of hologram pixels
Phase = fileData[2]  # Phase data of hologram pixels

if menu is 1:
    # MENU OPERATED RUNTIME
    if int(loop_var[0]) is 0:
        # SINGLE MODE

        F = MonochromaticField(wavelength=sim_var[1] * nm, extent_x=sim_var[2] * mm, extent_y=sim_var[3] * mm,
                               Nx=size * sim_var[0], Ny=size * sim_var[0], intensity=sim_var[4])
        F.phase_sampling(Phase, w0=sim_var[6] / 1000 * mm, size=size, sampling=int(sim_var[0]), threshold=sim_var[7],
                         amp_mod=sim_var[8],
                         steps=int(sim_var[9]), extended=int(sim_var[10]), verbose=bool(sim_var[11]))
        F.spherical_wave(z0=sim_var[5] * cm)
        F.integrate_intensity()

        rgb1 = F.compute_colors_at(sim_var[5] * cm)
        F.plot(rgb1, xlim=[-F.extent_x / 2 * 1000, F.extent_x / 2 * 1000],
               ylim=[-F.extent_y / 2 * 1000, F.extent_y / 2 * 1000])
        data = F.data_analysis(size, 1, export=True)
    elif int(loop_var[0]) is 1:
        # LOOP MODE

        date = datetime.date(datetime.now())
        time = datetime.time(datetime.now())
        time = time.strftime('%H-%M-%S')
        try:
            os.mkdir('./{}'.format(date))
        except OSError as error:
            print('Directory already exists (Ignore)')

        datafile = open('./{}/statistics_{}.txt'.format(date, time), 'w')
        datastring = "{}\t{}\t{}\t{}\n".format('variable', 'diff. efficiency', 'contrast', 'speckle noise')
        datafile.write(datastring)

        x = loop_var[2]
        while (x <= loop_var[3]):
            sim_var[loop_mapping[int(loop_var[1]) - 1]] = x
            F = MonochromaticField(wavelength=sim_var[1] * nm, extent_x=sim_var[2] * mm, extent_y=sim_var[3] * mm,
                                   Nx=size * sim_var[0], Ny=size * sim_var[0], intensity=sim_var[4])
            F.phase_sampling(Phase, w0=sim_var[6] / 1000 * mm, size=size, sampling=int(sim_var[0]), threshold=sim_var[7],
                             amp_mod=sim_var[8],
                             steps=int(sim_var[9]), extended=int(sim_var[10]), verbose=bool(sim_var[11]))
            F.spherical_wave(z0=sim_var[5] * cm)
            F.integrate_intensity()

            rgb1 = F.compute_colors_at(sim_var[5] * cm)
            plotpath = './{}/{}={}_{}.png'.format(date, sim_names[loop_mapping[int(loop_var[1]) - 1]], x, time)
            F.plot(rgb1, xlim=[-F.extent_x / 2 * 1000, F.extent_x / 2 * 1000],
                   ylim=[-F.extent_y / 2 * 1000, F.extent_y / 2 * 1000], export=plotpath)
            data = F.data_analysis(size, 1, export=True)
            datastring = "{}\t{}\t{}\t{}\n".format(x, data[0], data[1], data[2])

            datafile.write(datastring)
            x += loop_var[4]
        datafile.close()

"""#<---- REMOVE FOR MANUAL MODE
    # USAGE EXAMPLE / MANUAL MODE
    # VARIABLES REQUIRED TO BE REPLACED WITH VALUES ARE IN ALL CAPS
    # FOR LOOP IMPLEMENTATION LOOK BELOW

    F = MonochromaticField(wavelength= WAVELENGTH * nm, extent_x= SIZE IN X AXIS * mm, extent_y= SIZE IN Y AXIS * mm,
                           Nx=size * SUPERSAMPLING, Ny=size * SUPERSAMPLING, intensity= INTENSITY(0.1 DEFAULT))
    F.phase_sampling(Phase (LOADED FROM FILE), w0= BEAM WIDTH * mm, size=size (LOADED FROM FILE),
                    sampling= SUPERSAMPLING, threshold=(0-1 VALUE), amp_mod=(0-1 VALUE), steps=(0-X VALUE),
                    extended=(0-X VALUE), verbose=(0 OR 1 VALUE))
    F.spherical_wave(DISTANCE * cm)
    F.integrate_intensity()

    rgb1 = F.compute_colors_at(DISTANCE * cm)
    F.plot(rgb1, xlim=[-F.extent_x / 2 * 1000, F.extent_x / 2 * 1000],
           ylim=[-F.extent_y / 2 * 1000, F.extent_y / 2 * 1000], export='./manual_mode.png')
    data = F.data_analysis(size, 1, export=True)

    datafile = open('./manual_mode.txt', 'w')
    datastring = "{}\t{}\t{}\t{}\n".format('variable', 'diff. efficiency', 'contrast', 'speckle noise')
    datafile.write(datastring)
    datastring = "{}\t{}\t{}\t{}\n".format(x, data[0], data[1], data[2])
    datafile.write(datastring)

"""  # <---- REMOVE FOR MANUAL MODE

# LOOP
"""
    # EXAMPLE OF A LOOP
    # VARIABLES REQUIRED TO BE REPLACED WITH VALUES ARE IN ALL CAPS
    # STARTING, LAST AND STEP VALUES FOR THE LOOP NEED TO BE SET WHILE LOOPED VARIABLE NEEDS TO BE REPLACED BY x

    os.mkdir('./loop_results')
    datafile = open('./loop_results/manual_loop_mode.txt', 'w')
    datastring = "{}\t{}\t{}\t{}\n".format('variable', 'diff. efficiency', 'contrast', 'speckle noise')
    datafile.write(datastring)
    
x = STARTING VALUE
while (x <= LAST VALUE):
    F = MonochromaticField(wavelength= WAVELENGTH * nm, extent_x= SIZE IN X AXIS * mm, extent_y= SIZE IN Y AXIS * mm,
                           Nx=size * SUPERSAMPLING, Ny=size * SUPERSAMPLING, intensity= INTENSITY(0.1 DEFAULT))
    F.phase_sampling(Phase (LOADED FROM FILE), w0= BEAM WIDTH * mm, size=size (LOADED FROM FILE),
                    sampling= SUPERSAMPLING, threshold=(0-1 VALUE), amp_mod=(0-1 VALUE), steps=(0-X VALUE),
                    extended=(0-X VALUE), verbose=(0 OR 1 VALUE))
    F.spherical_wave(DISTANCE * cm)
    F.integrate_intensity()

    rgb1 = F.compute_colors_at(100 * cm)
    plotpath = './loop_results/variable={}.png'.format(x)
    F.plot(rgb1, xlim=[-F.extent_x / 2 * 1000, F.extent_x / 2 * 1000],
           ylim=[-F.extent_y / 2 * 1000, F.extent_y / 2 * 1000], export=plotpath)
    data = F.data_analysis(size, 1, export=True)
    datastring = "{}\t{}\t{}\t{}\n".format(x, data[0], data[1], data[2])

    datafile.write(datastring)
    x += LOOP STEP
datafile.close()
"""
