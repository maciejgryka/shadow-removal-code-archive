from experiment_common import *

def write_float_list_as_csv(float_list, file_name):
    s = ''.join(["%1.10f," % item for item in float_list])
    f = open(file_name, 'w')
    f.write(s)

def evaluate(test_dir, result_dir, image_names):
    # evaluate the results
    outcomes = {}
    outcomes['intensity'] = []
    outcomes['gradient_orientation'] = []
    outcomes['gradient_xy'] = []
    
    for im in image_names:
        if os.path.isfile(os.path.join(result_dir, ''.join([im, '_unshad.png']))):
            mask = mask_gt(test_dir, im)
            gt = noshad_gt(test_dir, im)
            gt_grad = gradient_orientation_image(gt)
            gt_dx, gt_dy = gradient_xy(gt)
            result = unshad_result(result_dir, im)
            result_grad = gradient_orientation_image(result)
            result_dx, result_dy = gradient_xy(result)
            # mask the arrays to only take into account pixels in shadow
            gt = gt[mask > 0]
            gt_grad = gt_grad[mask > 0]
            gt_dx = gt_dx[mask > 0]
            gt_dy = gt_dy[mask > 0]
            result = result[mask > 0]
            result_grad = result_grad[mask > 0]
            result_dx = result_dx[mask > 0]
            result_dy = result_dy[mask > 0]
            outcomes['intensity'].append(msd(result, gt))
            outcomes['gradient_orientation'].append(msd(result_grad, gt_grad))
            outcomes['gradient_xy'].append(msd(result_dx, gt_dx) + msd(result_dy, gt_dy))


    write_float_list_as_csv(outcomes['intensity'],
        os.path.join(result_dir, 'outcomes_intensity.csv'))
    write_float_list_as_csv(outcomes['gradient_orientation'],
        os.path.join(result_dir, 'outcomes_gradient_orientation.csv'))
    write_float_list_as_csv(outcomes['gradient_xy'],
        os.path.join(result_dir, 'outcomes_gradient_xy.csv'))
    
    return outcomes