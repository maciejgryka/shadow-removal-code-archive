import os
import sys
import re
import pickle
import random

import numpy as np

import experiment_common as xc
# from evaluate import *

OPTION_FILE = "prem_options.txt"


def test_and_evaluate(test_options_dict, iter_index=0):
    out_dir = test_options_dict['results_dir']
    xc.create_dir(out_dir)
    test_options_file = os.path.join(out_dir, 'test_options.txt')
    test_options_dict["iter_index"] = iter_index
    # test
    xc.pre_test_tasks(test_options_dict)
    print 'testing with %s...\n' % test_options_file
    xc.run_exec(test_options_dict["prem_exec"], [test_options_file])
    # post-test processing
    xc.post_test_tasks(test_options_dict);
    # # evaluate the results
    # outcomes = evaluate(test_options_dict['image_folder'], out_dir, test_names)
    # # save the results
    # out_file = open(os.path.join(out_dir, 'outcomes.pkl'), 'w')
    # pickle.dump(outcomes, out_file)
    # out_file.close()

    if iter_index < int(test_options_dict["n_iters"]) - 1:
        test_and_evaluate(test_options_dict, iter_index=iter_index+1)


grid_offset_x = [0, 2, 4, 6, 8, 10, 12, 14]
grid_offset_y = [0, 2, 4, 6, 8, 10, 12, 14]


def run(iter_index=0):
    # DBG: fix the seed for predictable shuffling
    dbg_seed = 1
    random.seed(dbg_seed)

    options = xc.read_options(OPTION_FILE)

    target_path = options["target_path"]
    n_training_images = options["n_training_images"]

    deblock_opt = options["deblock_opt_file_source"]

    training_dir = options["training_dir"]
    training_images = xc.get_file_names(training_dir, xc.is_shad)
    training_names = xc.separate_image_names(training_images)
    
    test_dir = options["test_dir"]
    test_images = xc.get_file_names(test_dir, xc.is_shad)
    test_names = xc.separate_image_names(test_images)
    
    output_dir = options["output_dir"]
    image_list_name = options["image_list_name"]
    
    outcomes = {}

    prem_exec = options["prem_exec"]

    # create a directory to store the results
    size_dir = options["size_dir"]
    xc.create_dir(size_dir)

    data_dir = os.path.join(size_dir, 'data')
    xc.create_dir(data_dir)

    if int(options["train"]):
        # take n_training_images images from the training set
        first_im = 0;
        # randomly subsample training_names
        if int(options["random_training_subsample"]) == 1:
            sub_training_names = random.sample(training_names, int(n_training_images))
        else:
            sub_training_names = training_names[:int(n_training_images)]
        # write image names to a file
        image_list_file_training = options["image_list_file_training"]
        xc.write_iterable_to_file(sub_training_names, image_list_file_training)
    
    image_list_file_test = options["image_list_file_test"]
    results_dir = options["results_dir"]

    if int(options['train']):
        training_options_dict = {}
        training_options_dict = xc.copy_options(xc.OPTIONS_COMMON, options, training_options_dict)
        training_options_dict = xc.copy_options(xc.OPTIONS_TRAINING, options, training_options_dict)
        training_options_dict["image_folder"] = training_dir
        training_options_dict['image_list_file'] = image_list_file_training
        training_options_dict["train"] = 1
        training_options_dict["test"] = 0
        
        # write training options to file
        training_options_file = os.path.join(size_dir, 'training_options.txt')
        xc.write_options(options_file = training_options_file, options_dict = training_options_dict)
        # train on this subset
        print 'training with %s...\n' % training_options_file
        xc.pre_train_tasks(training_options_dict)
        xc.run_exec(prem_exec, [training_options_file])

    if int(options['test']):
        # write test options to file
        test_options_dict = {}
        test_options_dict = xc.copy_options(xc.OPTIONS_COMMON, options, test_options_dict)
        test_options_dict = xc.copy_options(xc.OPTIONS_TEST, options, test_options_dict)
        test_options_dict["training_image_folder"] = training_dir
        test_options_dict["image_folder"] = test_options_dict["image_folder_original"] = test_dir
        test_options_dict['image_list_file'] = test_options_dict['image_list_file_original'] = image_list_file_test
        test_options_dict["train"] = 0
        test_options_dict["test"] = 1
        
        # # read GS postprocessing options
        # gs_opts = open(deblock_opt, 'r').read()
        # # append outDir option
        # gs_opts += "outDir = %s" % (results_dir)
        # gs_opts_file = open(os.path.join(data_dir, "deblock-opt.txt"), "w")
        # gs_opts_file.write(gs_opts);
        # gs_opts_file.close()

        if int(test_options_dict["grid_offset"]):
            for gox in grid_offset_x:
                for goy in grid_offset_y:
                    test_options_dict["results_dir"] = os.path.join(results_dir, "%s_%s"%(gox, goy))
                    test_options_dict["grid_offset_x"] = gox
                    test_options_dict["grid_offset_y"] = goy

                    test_and_evaluate(test_options_dict)
        else:
            test_and_evaluate(test_options_dict, 0)

if __name__ == "__main__":
    run()
