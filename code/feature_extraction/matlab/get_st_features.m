function get_st_features(image_set, args);

    if nargin<2; args=struct([]); end
   
    if ~isdir(args.save_dir)
        fprintf('Creating dir at %s\n', args.save_dir)
        mkdir(args.save_dir)
    end
    
    if isfield(args, 'downsample_factor'); downsample_factor=args.downsample_factor; else; downsample_factor=1.0; end
    if isfield(args, 'batch_size'); batch_size=args.batch_size; else; batch_size=100; end
    if isfield(args, 'overwrite'); overwrite = args.overwrite; else; overwrite = 0; end
    overwrite = boolean(overwrite);
    if isfield(args, 'debug'); debug = args.debug; else; debug = 0; end 
    if isfield(args, 'grayscale'); grayscale = args.grayscale; else; grayscale = 0; end 
    
    debug = boolean(debug);
    grayscale = boolean(grayscale);
    
    addpath(genpath(fullfile(args.sketchtokens_dir)));  
    addpath(genpath(fullfile(args.toolbox_dir)));
    
    
    stDir = args.sketchtokens_dir;
    model_fn = fullfile(stDir, 'models/forest/modelSmall.mat');
    fprintf('Loading model from %s...\n', model_fn)
    model = load(model_fn);
    model = model.model;
   
   
    fprintf('Reading image brick from %s...\n', args.image_filename)
    I = h5read(args.image_filename, '/stimuli');
    fprintf('examples of raw image values:\n')
    disp(I(1,1,:,1:3))
    
    if args.grayscale & size(I,3)>1
        % convert this image into grayscale
        fprintf('converting the image to grayscale before processing\n')
        I = I(:,:,1,:)*0.2126729 + I(:,:,2,:)*0.7151522 + I(:,:,3,:)*0.0721750;
        disp(size(I))
    end
    
    
    [h, w, c, n_total_ims] = size(I);
    if downsample_factor>1
        h = ceil(h/downsample_factor);
        w = ceil(w/downsample_factor);
        fprintf('After downsampling, image size will be [%d by %d]\n', h, w)
    end
    if c==1
        I = repmat(I,[1,1,3,1]);
    end

    fprintf('examples of preprocessed image values:\n')
    disp(I(1,1,:,1:3))
    
    n_features = 151;
    
    if args.grayscale
        save_fn = fullfile(args.save_dir, sprintf('%s_features_grayscale_%d.h5py', image_set, h)); 
        save_fn_edges = fullfile(args.save_dir, sprintf('%s_edges_grayscale_%d.h5py', image_set, h));  
    else
        save_fn = fullfile(args.save_dir, sprintf('%s_features_%d.h5py', image_set, h)); 
        save_fn_edges = fullfile(args.save_dir, sprintf('%s_edges_%d.h5py', image_set, h)); 
    end
    
    fprintf('Will write features to %s...\n',save_fn);    
    if exist(save_fn,'file')
        if overwrite
            fprintf('File exists, removing it now...\n')
            unix(char(sprintf('rm %s',save_fn)));
        else
            error('File already exists. To overwrite it set overwrite=True.');
        end
    end
    
    fprintf('Will write final edge maps to %s...\n',save_fn_edges);
    if exist(save_fn_edges,'file')
        if overwrite
            fprintf('File exists, removing it now...\n')
            unix(char(sprintf('rm %s',save_fn_edges)));
        else
            error('File already exists. To overwrite it set overwrite=True.');
        end
    end
    
    % Note: when this array gets loaded in python later, the order gets switched so it will be:
    % [n_features x w x h x n_images]
    h5create(save_fn,'/features',[Inf, h, w, n_features], "Chunksize",[batch_size, h, w, n_features]);
    h5create(save_fn_edges,'/features',[Inf, h, w], "Chunksize",[batch_size, h, w]);
   
    n_batches = ceil(n_total_ims/batch_size);
    for bb = 1:n_batches
        if debug && bb>2
            break
        end
        batch_inds = [(bb-1)*batch_size+1:min([bb*batch_size, n_total_ims])];
        image_batch = I(:,:,:,batch_inds);
        [features_batch, edges_batch] = get_features_batch(image_batch);
        tic
        fprintf('    Writing batch %d of %d...\n', bb, n_batches);
        h5write(save_fn, '/features', features_batch, [batch_inds(1),1,1,1], size(features_batch));
        h5write(save_fn_edges, '/features', edges_batch, [batch_inds(1),1,1], size(edges_batch));
        toc
    end
    
    function [features_batch, edges_batch] = get_features_batch(image_batch)

        n_ims_batch = size(image_batch, 4);
        if debug
            n_ims_do = 2;
        else
            n_ims_do = n_ims_batch;
        end

        features_batch = zeros(n_ims_batch, h, w, n_features);
        edges_batch = zeros(n_ims_batch, h, w);
        tic
        fprintf('    Extracting features...\n');
        for ii = 1:n_ims_do

            image = squeeze(image_batch(:,:,:,ii));
            if downsample_factor>1
                image = imresize(image, [h,w])
            end

            features = stDetect(image, model);
            edges = stToEdges(features, 1, 1);
            
            features_batch(ii,:,:,:) = features;
            edges_batch(ii,:,:) = edges;
            

        end
        toc
        
    end


end
        
    