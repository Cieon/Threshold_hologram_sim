import easygui

# Variable information for help in the menu
var_info=[
    "Parameter ssampling refers to the amount of pixels supersampling each point of phase which has to be written, ex. ssampling=16 results in a cell of 16x16.",
    "Wavelength means the wavelength of light used for simulation in nanometers, used values from the visible range (400-700 nm).",
    "Variable extent_x refers to the size of the entire pattern of the written hologram in the x axis in millimeters.",
    "Variable extent_y refers to the size of the entire pattern of the written hologram in the y axis in millimeters.",
    "Intesity of the pattern means the max amplitude in each written pixel and overall controls the brightness or the resulting image",
    "Distance refers to the distance of reconstruction of the hologram in centimeters.",
    "Width of the writing Gaussian beam in micrometers in each base cell of the pattern. Size of each cell equals to extent_x / hologram size.",
    "Threshold of a cell refers to a value in 0-1 range above which the Gaussian beam will write a pattern.",
    "Parameter amp_mod sets the modulation of the pattern in each cell, 0 for binary phase modulation, 1 for binary amplitude modulation and values between for combination of both modulations.",
    "Steps refer to so called superpixels in the writing cell surrounding the above threshold written pattern and dampening the change between the modulation. Only to be used for amp_mod = 1.",
    "Using the extended parameter allows the neighboring cells to overlap for the /extended/ amount of pixels. Only to be used for amp_mod = 1.",
    "Verbose equal to 1 shows plots of the amplitude and phase of a base writing cell and the entire pattern.",
    "Confirm the displayed parameters."
]

# Loop settings information for help in the menu
loop_info=[
    "Whether the loop is enabled or disabled.",
    "Parameter changing during the loop: \n1-ssampling\t2-wavelength [nm]\t3-distance [cm]\t4-w0 [um]'\n"
    "5-threshold\t6-amp_mod\t\t\t7-steps\t\t\t8-extended",
    "Starting value of the loop for the set parameter.",
    "Final value of the loop for the set parameter",
    "Step of the loop between the starting and final values.",
    "Confirm the chosen loop parameters."
]


def print_welcome():
    # Welcome procedure
    print('### Welcome to the threshold medium diffraction simulator ###')
    print('For manual mode, edit the script for parameters and run with 1st argument as 0')


def print_main_menu():
    # Main menu text
    print('1 -- File selection')
    print('2 -- Adjust parameters of simulation')
    print('3 -- Loop settings')
    print('4 -- Execute simulation')
    print('5 -- Exit')


def program_help():
    # Main menu help
    print("This program is used for analysis of hologram simulations in a medium with threshold intensity writing "
          "using supersampling of each phase pixel.\nDiffraction is calculated with the angular spectrum "
          "method.\nAccepted files are in pairs of float numbers representing real and imaginary value of each "
          "hologram pixel.\nExplanations for the meaning of each parameter are included in submenu 2.")


def invalid_option(options):
    # Print range of valid options
    message = 'Invalid option. Please enter a number between 1 and {}.'.format(options)
    print(message)


def file_selector():
    # File selector to read
    print('Select file to read')
    path = easygui.fileopenbox()
    print('Selected file:' + path)
    return path


def adjust_variables(names, values):
    # Print variable names and their values, then the confirm option
    for x in range(12):
        print('{} \t -- \t {} : {}'.format(x + 1, names[x], values[x]))
    print('13 \t -- \t Confirm')
    # Ask for input
    option = input('Enter option or type ?x for parameter help: ')
    if '?' in option:
        # Check for help request and print info
        option.replace('?', '')
        option = int(option[1:])    # Check which variable
        if option < 0 or option > 13:
            invalid_option(13)  # Check range
        else:
            # Print info
            print(var_info[option-1])
            return values, option
    else:
        # Handle parameter assignment
        option = int(option)
        if option < 0 or option > 13:
            # Wrong option
            invalid_option(13)
            return values, option
        elif option is 13:
            # Return to main menu on confirmation of all parameters
            return values, option
        else:
            # Assigning new value to the list of values
            print('Change the parameter {}'.format(names[option - 1]))
            values[option - 1] = float(input('Enter new value: '))
            return values, option


def loop_setup(names, values):
    # Procedure the same as with adjust_variables(names, values)
    for x in range(5):
        print('{} \t -- \t {} : {}'.format(x + 1, names[x], values[x]))
    print('6 \t -- \t Confirm')
    option = input('Enter option or type ?x for parameter help: ')
    if '?' in option:
        option.replace('?', '')
        option = int(option[1:])
        if option < 0 or option > 6:
            invalid_option(5)
        else:
            print(loop_info[option-1])
            return values, option
    else:
        option = int(option)
        if option < 0 or option > 5:
            invalid_option(5)
            return values, option
        elif option is 6:
            return values, option
        else:
            print('Change the parameter {}'.format(names[option - 1]))
            if option is 2:
                print(loop_info[1])
            values[option - 1] = float(input('Enter new value: '))
            return values, option


def run_menu(file_path, sim_names, sim_var, loop_names, loop_var):
    """
        Prints out an interactable menu allowing user to assign values and run the simulation
        Parameters:
            (I) file_path: path to the hologram file
            (I) sim_names: names of the simulation parameters
            (I) sim_var: values of the simulation parameters
            (I) loop_names: names of the loop parameters
            (I) loop_var: values of the loop parameters
            (O) returns file_path, sim_var and loop_var with new or unchanged values
    """

    menu = 1    # Menu running when 1
    var_list = sim_var  # Copy
    loop_list = loop_var    # Copy
    while menu:
        print_main_menu()
        option = input('Choose an option or type ?help: ')  # User option selection
        if "?help" in option:
            # Print help
            program_help()
            continue
        else:
            # Handle choice
            option = int(option)
        if option == 1:
            # Select file
            file_path = file_selector()
        elif option == 2:
            # Changing variables
            adjusting = True
            while adjusting is True:
                var_list, var_input = adjust_variables(sim_names, var_list)
                if var_input is 13:
                    # Confirmation of variables
                    adjusting = False
        elif option == 3:
            # Changing loop setup
            adjusting = True
            while adjusting is True:
                loop_list , loop_input = loop_setup(loop_names, loop_list)
                if loop_input is 6:
                    # Confirmation of loop setup
                    adjusting = False
        elif option == 4:
            menu = 0    # Shut down the loop
            return file_path, var_list, loop_list   # Return parameters and start simulation
        elif option == 5:
            # Exiting program
            print('Exiting program')
            exit()
        else:
            # Invalid choice
            invalid_option(5)
