import sys

with open(sys.argv[1],'r') as f:
    lines = f.readlines()
    prev_line = None
    prev_region_single_voxel = False
    n_regions_found = 0
    n_regions_found_no_single_voxels = 0
    n_regions_suggested = 0

    for line in lines:
    	if 'number of regions in target' in line:
    		n_regions  = int(line.strip('number of regions in target'))
    		n_regions_found += n_regions
    		if not prev_region_single_voxel:
    			n_regions_found_no_single_voxels += n_regions
        elif 'region' in line:
        	if line.rstrip().split()[5].rstrip(']') == '1':
        		prev_region_single_voxel = True
        	else:
        		prev_region_single_voxel = False
        		n_regions_suggested += 1


    print 'n_regions_found', n_regions_found
    print 'n_regions_found_no_single_voxels', n_regions_found_no_single_voxels
    print 'n_regions_suggested', n_regions_suggested
