# place holder for the procdeure to excute broadband efc

def hi(mode: str) -> None:
    """
    Main function to run the CAPyBARA simulation.

    Parameters
    ----------
    mode : str
        Observation mode ('mono' for monochromatic, 'broadband' for broadband).
        Jacobian 
    """
    print('Run CAPYBARA')

    if mode == 'broadband':

        save = True 
        print('Try Broadband')

        experiment = 'BroadbandEFC'

        param_rst['jacobian'] = param_rst['path']+param_rst['jacobian']

        # Assuming central_wvl is a numpy array and aberration_ptv is scalar or array
        param_rst['central_wvl'] = np.array(param_rst['central_wvl'])
        # If aberration_ptv is a scalar, convert it to an array with the same shape as central_wvl
        param_rst['aberration_ptv'] = param_rst['aberration_ptv'] * param_rst['central_wvl'] * 1e-9

        CAPyBARA_list = []
        aberration_class_list = []

        for i in range(len(param_rst['central_wvl'])):
            param = param_rst.copy()
            param['central_wvl'] = np.array(param_rst['central_wvl'][i])
            param['aberration_ptv'] = np.array(param_rst['aberration_ptv'][i])

            # Initialise simulation

            CAPyBARA = CAPyBARAsim(param)

            CAPyBARA.get_grid()

            if i == 0: 
                print('Calculating influence function')
                influence_function = make_xinetics_influence_functions(CAPyBARA.pupil_grid, CAPyBARA.param['num_actuator'], CAPyBARA.param['actuator_spacing'])

            CAPyBARA.get_system(influence_function)
            CAPyBARA.get_prop()
            CAPyBARA.set_aberration()

            CAPyBARA.get_reference_image(wvl=param['central_wvl']*1e-9, check=False)

            # may be we need them individually? I do not know but it seems like no, since it contains methods depending on incoming wf and wvl
            # individually defined by each function but the aberration cube? 
            # TODO - maybe look into this later and try to fix this
            aberration_class = rst_aberration.CAPyBARAaberration(sim=CAPyBARA)
            aberration_class.set_zernike_basis(num_mode=21)

            step0_components_array = aberration_class.extract_component(CAPyBARA.aberration_func(CAPyBARA.wf_ref)) 
            field0 = Field(np.dot(step0_components_array, aberration_class.zernike_basis), CAPyBARA.pupil_grid)

            # param['num_iteration'] = 5

            updated_wf, updated_zernike_coeff, n_aberration = aberration_class.apply_perturbation_to_wavefront(step0_components_array, param['central_wvl'], seed=1)
            n_zernike_coeff, n_field, n_aberration = aberration_class.track_zernike_component(step0_components_array,param['central_wvl'], field0)
            aberration_class.get_aberration_data_cube(n_field)

            print(f'Shape of the n_field object: {np.shape(n_field)}')

            CAPyBARA_list.append(CAPyBARA)
            aberration_class_list.append(aberration_class)

        #%% EFC

        efc_exp = EFieldConjugation(CAPyBARA_list, aberration_class_list, wvl=param_rst['central_wvl'])

        actuator_list, e_field_list, img_list, wf_lyot_list, wf_residual_list  = efc_exp.control(wvl=param_rst['central_wvl'])

        # Get the average contrast

        # print(f'Checking the lenght of the image list{img_list.shape}')
        average_contrast = rst_func.get_average_contrast(CAPyBARA_list,img_list)
        
        print(f'Contrast: {average_contrast}')
        # print(f'Type of wf_residual_list {img_list[0][0]}, shape of wf_residual_list{img_list.shape}')

        wvl0 = param_rst['central_wvl'][0]
        wvl1 = param_rst['central_wvl'][1]

        print(f"Length of img_list: {len(img_list)}")
        print(f"Length of img_list[0]: {len(img_list[0])}")

        plot=False

        if plot is True: 
            plotting.plt_focal_image(CAPyBARA_list[0], img_list[0][0], title=f'Iteration 0 at wvl = {wvl0} (Image)', cmap= 'inferno')
            plotting.plt_focal_image(CAPyBARA_list[1], img_list[0][1], title=f'Iteration 0 at wvl = {wvl1} (Image)', cmap= 'inferno')

            plotting.plt_focal_image(CAPyBARA_list[0], img_list[-1][0], title=f'Last iteration at wvl = {wvl0} (Image)(Image)', cmap= 'inferno')
            plotting.plt_focal_image(CAPyBARA_list[1], img_list[-1][1], title=f'Last iteration at wvl = {wvl1} (Image)(Image)', cmap= 'inferno')

        # TODO - Write an automatic pipeline to offload the data
        print('Broadband Observing Sequence')

        # wvl for the EFC is different from the science acquistion

        # OS_class = ObservingSequence(CAPyBARA, aberration_class)

        # ref_star_list = OS_class.accquisition_loop(wvl=[param_rst['central_wvl']], aberration_sequence=n_field[2:3], last_dm_command=actuator_list[-1])

        if save is True: 

            date = datetime.today().strftime('%Y-%m-%d')
            path = path + date + '_' + efc_exp.name
            np.savetxt(os.path.join(path+'contrast.txt'), average_contrast)

            for i in range (param_rst['num_iteration']):
                print(f'What iteration saving? {i}')
                for j in range (len(param_rst['central_wvl'])):
                    # save the reference
                    print('Which wvl saving?', param_rst['central_wvl'][j])
                    CAPyBARA_list[j].get_reference_image(wvl=param_rst['central_wvl'][j]*1e-9, check=False)
                    
                    ref_img = CAPyBARA_list[j].ref_img.shaped
                    utils.write2fits(ref_img, key='direct', wvl=param_rst['central_wvl'][j],path=os.path.join(path, f'iteration_{i:04n}'))

                    corona_img = img_list[i][j].shaped
                    # save the coronographic image # corona_img = img_list[i][j]
                    utils.write2fits(corona_img, key='sci', wvl=param_rst['central_wvl'][j], path=os.path.join(path, f'iteration_{i:04n}'))

                # Get the dm surface
                CAPyBARA_list[j].apply_actuators(actuator_list[i])
                
                dm1_surface = CAPyBARA_list[j].dm1.surface.shaped
                utils.write2fits(dm1_surface, key='dm1_surface', wvl=param_rst['central_wvl'][j], path=os.path.join(path, f'iteration_{i:04n}'))

                dm2_surface = CAPyBARA_list[j].dm2.surface.shaped
                utils.write2fits(dm2_surface, key='dm2_surface', wvl=param_rst['central_wvl'][j], path=os.path.join(path, f'iteration_{i:04n}'))

                utils.write2fits(actuator_list[i], key='dm', wvl=param_rst['central_wvl'][j], path=os.path.join(path, f'iteration_{i:04n}'))

        print('End')



            # field_list = []

            # for i in range (len(n_field)):
            #     _feild = Field(n_field[i], CAPyBARA_list[0].pupil_grid).shaped
            #     field_list.append(_feild)

            # # obs_aberration_class_list.append(obs_aberration_class) aberration_class
            # import matplotlib.pyplot as plt
            # from matplotlib.animation import FuncAnimation

            # # Create a figure and axis for the animation
            # fig, ax = plt.subplots()

            # # Initialize the image plot
            # img = ax.imshow((field_list[0]*CAPyBARA_list[0].param['wvl']/(2*np.pi)), cmap='inferno')

            # # Add a colorbar linked to the image
            # cbar = fig.colorbar(img, ax=ax)

            # # Function to update the image for each frame in the animation
            # def update(frame):
            #     img.set_data((frame * CAPyBARA_list[0].param['wvl'] / (2 * np.pi)))
            #     return [img]

            # # Create the animation
            # ani = FuncAnimation(fig, update, frames=field_list, blit=True, repeat=False)

            # # Show the animation in a window
            # plt.show()