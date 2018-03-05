"""  Convert .fits files files into (45,45) .npy files.
Randomly shuffles the files and selects 20 per cent of the files for TestData.
Note: Indices of .npy and .fits are NOT the same.
"""


import numpy as np
import matplotlib.pylab as plt
import glob
from astropy.io import fits
# import SetPub
# SetPub.set_pub()


np.random.seed(1562)
randomArray =  np.random.randint(10000, size = 8)

Dir1 = '/home/nes/Desktop/ConvNetData/lens/'

Dir2 = ['data_of_lsst/'][0]

# 'lsst_noiseless_single/', 'lsst_noiseless_stack/'
# Only lensed and noiseless - 20k images each - single and stack images

# 'data_of_lsst/' single and stack lensed images go from 0 - 10k [for both 0/ and 1/],
# unlensed images from 0 - 10k [0/] and 10k-20k [1/]

# Dir3 = 'data_of_lsst/'
Dir3 = ['lsst_mocks_single/', 'lsst_mocks_stack/'][1]
Dir5 = ['0/', '1/'][0]  # 10k images each for lensed and unlensed


for lid in [0]:  # lensed or unlensed - trial classification b/w 2 labels.

    labels = [ 'lensed_', 'unlensed_'][lid]
    Dir4 = labels+'outputs/'


    fileIn = Dir1+Dir2+Dir3+Dir4+Dir5+'*gz'
    # fileIn = '/home/nes/Desktop/ConvNetData/lens/lsst_noiseless_stack/'+'*gz'
    print fileIn


    print 'FITS folder: ', fileIn

    fileInData = sorted(glob.glob(fileIn))

    if (len(fileInData) == 0): print 'ERROR: Empty folder'
    print 'number of files: ', len(fileInData)

    # alln = np.arange(len(fileInData))
    # np.random.shuffle(alln)

    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize = (10, 5))
    plt.suptitle( fileIn )

    count = 0
    for ind in randomArray:
        fileInPix = fileInData[ind]
        pixel = fits.open(fileInPix, memmap=True)

        # for i in range(numPlots):
        ax[ count/4, count%4].imshow( pixel[0].data , cmap=plt.get_cmap('gray'))
        ax[count / 4, count % 4].set_title(str(ind))

        count +=1
    plt.savefig( 'plots/'+ Dir2[:-1]+labels+'.png' )
plt.show()
