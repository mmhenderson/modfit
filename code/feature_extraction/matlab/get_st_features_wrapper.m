function get_st_features_wrapper(image_set, image_filename, save_dir, sketchtokens_dir, toolbox_dir, batch_size, grayscale, debug)

    args = struct;
    args.debug = debug;
    args.batch_size=10;
    args.overwrite=1;
    args.downsample_factor=1;
    
    args.save_dir=save_dir;
    args.sketchtokens_dir=sketchtokens_dir;
    args.toolbox_dir=toolbox_dir;
    args.image_filename=image_filename;
    args.grayscale=grayscale;
   
    fprintf('Inputs are:')
    disp(image_set)
    disp(args)
    
    get_st_features(image_set, args)

end