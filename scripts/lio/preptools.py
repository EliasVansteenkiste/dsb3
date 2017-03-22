import numpy as np
import dicom
import os
import scipy.ndimage


# Load the scans in given folder path
def load_scan(path, stop_before_pixels=False):
    slices = [dicom.read_file(path + '/' + s, stop_before_pixels=stop_before_pixels) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    # One metadata field is missing, the pixel size in the Z direction, which is the slice thickness.
    # Fortunately we can infer this, and we add this to the metadata.
    try:
        slice_thickness = slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
    except:
        slice_thickness = slices[0].SliceLocation - slices[1].SliceLocation

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def get_spacing(_slice):
    # Determine current pixel spacing
    spacing = map(float, ([_slice.SliceThickness] + _slice.PixelSpacing))
    spacing = np.array(list(spacing))

    # DennisSakva comment: For images with negative z axis I would also reverse the order of slices to preserve head is up orientation.
    # Or else the lungs will be upside down on the plots
    if spacing[0] < 0:
        spacing[0] = -spacing[0]
        flipped = True
    else: flipped = False
    return spacing, flipped


def resample(image, spacing, new_spacing=[1, 1, 1], fixed_size=None):

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order = 1)

    if fixed_size is None: return image, new_spacing

    img = np.zeros(fixed_size, np.int16)

    if image.shape[0] > img.shape[0]:
        xslice = slice(0, img.shape[0])
        l = image.shape[0]//2-img.shape[0]//2
        oxslice = slice(l, l+img.shape[0])
    else:
        l = img.shape[0] // 2 - image.shape[0] // 2
        xslice = slice(l, l + image.shape[0])
        oxslice = slice(0, image.shape[0])

    if image.shape[1] > img.shape[1]:
        yslice = slice(0, img.shape[1])
        l = image.shape[1]//2-img.shape[1]//2
        oyslice = slice(l, l+img.shape[1])
    else:
        l = img.shape[1] // 2 - image.shape[1] // 2
        yslice = slice(l, l + image.shape[1])
        oyslice = slice(0, image.shape[1])

    if image.shape[1] > img.shape[1]:
        zslice = slice(0, img.shape[1])
        l = image.shape[1]//2-img.shape[1]//2
        ozslice = slice(l, l+img.shape[1])
    else:
        l = img.shape[1] // 2 - image.shape[1] // 2
        zslice = slice(l, l + image.shape[1])
        ozslice = slice(0, image.shape[1])

    img[xslice, yslice, zslice] = image[oxslice, oyslice, ozslice]

    return img, new_spacing


def plot_3d(img, threshold=-300, cut=True, spacing=[1,1,1]):
    from mayavi import mlab
    import mayavi
    # img = img.transpose((0,2,1))
    # img = img.T[:,:, ::-1]

    mlab.clf()
    mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))

    # src = mlab.contour3d(img, contours=[theshold])
    src = mlab.pipeline.scalar_field(img)
    src.spacing = spacing
    src.update_image_data = True

    # blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
    if cut:
        voi = mlab.pipeline.extract_grid(src)
        voi.set(x_min=0, x_max=img.shape[0]-1,
                y_min=img.shape[1]//2, y_max=img.shape[1]-1,
                z_min=0, z_max=img.shape[2]-1)
    else: voi = src

    mlab.pipeline.iso_surface(voi, contours=[threshold,], color=(1, 1, 1))


    # thr = mlab.pipeline.threshold(src, low=1120)
    # cut_plane = mlab.pipeline.scalar_cut_plane(thr,
    #                   plane_orientation='y_axes',
    #                   colormap='black-white',
    #                   vmin=1400,
    #                   vmax=2600)
    # cut_plane.implicit_plane.origin = (136, 111.5, 82)
    # cut_plane.implicit_plane.widget.enabled = False
    # cut_plane2 = mlab.pipeline.scalar_cut_plane(thr,
    #                                             plane_orientation='z_axes',
    #                                             colormap='black-white',
    #                                             vmin=1400,
    #                                             vmax=2600)
    # cut_plane2.implicit_plane.origin = (136, 111.5, 82)
    # cut_plane2.implicit_plane.widget.enabled = False
    # outer = mlab.pipeline.iso_surface(src, contours=[-500, ],
    #                                     color=(0.8, 0.7, 0.6))
    # mlab.roll(180)
    mlab.show()



    # src = mlab.pipeline.scalar_field(img)
    # # Our data is not equally spaced in all directions:
    # # src.spacing = [1, 1, 1]
    # src.update_image_data = True
    # #
    # # Extract some inner structures: the ventricles and the inter-hemisphere
    # # fibers. We define a volume of interest (VOI) that restricts the
    # # iso-surfaces to the inner of the brain. We do this with the ExtractGrid
    # # filter.
    # blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
    # voi = mayavi.tools.pipeline.extract_grid(blur)
    # voi.set(x_min=125, x_max=173, y_min=92, y_max=125, z_min=34, z_max=75)
    #
    # mlab.pipeline.iso_surface(voi, contours=[-500], colormap='Spectral')
    #
    # # Add two cut planes to show the raw MRI data. We use a threshold filter
    # # to remove cut the planes outside the brain.
    # # thr = mlab.pipeline.threshold(src, low=1120)
    # # cut_plane = mlab.pipeline.scalar_cut_plane(thr,
    # #                                            plane_orientation='y_axes',
    # #                                            colormap='black-white',
    # #                                            vmin=1400,
    # #                                            vmax=2600)
    # # cut_plane.implicit_plane.origin = (136, 111.5, 82)
    # # cut_plane.implicit_plane.widget.enabled = False
    # #
    # # cut_plane2 = mlab.pipeline.scalar_cut_plane(thr,
    # #                                             plane_orientation='z_axes',
    # #                                             colormap='black-white',
    # #                                             vmin=1400,
    # #                                             vmax=2600)
    # # cut_plane2.implicit_plane.origin = (136, 111.5, 82)
    # # cut_plane2.implicit_plane.widget.enabled = False
    # #
    # # # Extract two views of the outside surface. We need to define VOIs in
    # # # order to leave out a cut in the head.
    # # voi2 = mayavi.tools.pipeline.extract_grid(src)
    # # voi2.set(y_min=112)
    # # outer = mlab.pipeline.iso_surface(voi2, contours=[-500, ],
    # #                                   color=(0.8, 0.7, 0.6))
    # #
    # # voi3 = mayavi.tools.pipeline.extract_grid(src)
    # # voi3.set(y_max=112, z_max=53)
    # # outer3 = mlab.pipeline.iso_surface(voi3, contours=[-500, ],
    # #                                    color=(0.8, 0.7, 0.6))
    #
    # mlab.view(-125, 54, 326, (145.5, 138, 66.5))
    # mlab.roll(-175)
    #
    # mlab.show()

def plot_3d_mpl(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    p = p[:, :, ::-1]

    from skimage import measure
    verts, faces = measure.marching_cubes(p, threshold)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
