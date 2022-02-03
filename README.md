# Threshold_hologram_sim
This program is used for analysis of hologram simulations in a medium with threshold intensity writing using supersampling of each phase pixel. Diffraction is calculated with the angular spectrum method. Accepted files are in pairs of float numbers representing real and imaginary value of each hologram pixel. Explanations for the meaning of each parameter are included in submenu 2.

Requiremets:
- Python 3.7
- diffractsim 1.3.0 - pip install diffractsim==1.3.0
- easygui - pip install easygui
- appropriate hologram file - binary file containing float values representing real and imaginary values of each of the pixels: Re, Im, Re, Im, .... The saved hologram has to be a square. An example file is included.

Usage:
- python diffsim.py - menu controlled mode
- python diffsim.py 0 - manual mode, controlled through edition of the diffsim.py script, explanations included in the file.

Output:
- in single simulation mode outputs the statistics into the terminal and saves the resulting field into last_sim.png
- in loop mode saves all results into a date directory with images in format variable=_value_start-time.png and statistics_start-time.txt
- manual mode saves into a location specified by the user.

Credits:
- https://github.com/rafael-fuente/diffractsim - basis for the simulation part.
