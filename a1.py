#CS 543 a1
import sys
import argparse
import time
from PIL import Image
import numpy as np


show_progress = False

def evaluate(imageA_array, imageB_array, x_offset, y_offset):
    total = 0.0
    for i in range(max(0,y_offset), min(y_offset+len(imageB_array),len(imageA_array))):
        for j in range(max(0,x_offset), min(x_offset+len(imageB_array[0]),len(imageA_array[0]))):
            total += imageA_array[i][j] * imageB_array[i-y_offset][j-x_offset]
            #if( i >= y_offset and j >= x_offset and (i-y_offset) < len(imageB) and (j-x_offset) < len(imageB[0])):
                #total += (int(imageA[i][j]) - int(imageB[i-y_offset][j-x_offset]))**2
                
    return total
    

def normalize(image_array):
    mean = np.mean(image_array)
    norm = np.linalg.norm(image_array)
    return (image_array - mean)/norm



def align(imageA, imageB, radius_x, radius_y, show_time, x_offset = None, y_offset = None):


    if(x_offset == None):
        x_offset = ((imageA.width - imageB.width)//2)-radius_x
    
    if(y_offset == None):
        y_offset = ((imageA.height - imageB.height)//2)-radius_y
    
    dimension_x = 2*radius_x+1
    dimension_y = 2*radius_y+1

    imageA_array = normalize(np.array(imageA))
    imageB_array = normalize(np.array(imageB))
    
    max_score = evaluate(imageA_array, imageB_array, x_offset, y_offset)
    best_x = x_offset
    best_y = y_offset


    
    for i in range(dimension_y):
        for j in range(dimension_x):

            if(show_progress and show_time):
                start_time = time.time()

            score = evaluate(imageA_array, imageB_array, x_offset+j, y_offset+i)

            if(show_progress and show_time):
                end_time = time.time()
                print("Time:"+str(end_time - start_time))

            #print(str(x_offset+j)+","+str(y_offset+i)+":"+str(score))
            if(score > max_score):
                max_score = score
                best_x = x_offset+j
                best_y = y_offset+i

                if(show_progress):
                    print("Current best alignment:"+str(best_x)+","+str(best_y)+" Score:"+str(score))
    
    return best_x, best_y


def align_multiscale(imageA, imageB, min_scale):

    imageA_half = imageA.resize((imageA.width//2, imageA.height//2))
    imageB_half = imageB.resize((imageB.width//2, imageB.height//2))

    if((imageA_half.width < min_scale) or (imageA_half.height < min_scale) or (imageB_half.width < min_scale) or (imageB_half.height < min_scale)):
        x_offset = ((imageA.width - imageB.width)//2)-1
        y_offset = ((imageA.height - imageB.height)//2)-1

    else:
        x_offset, y_offset = align_multiscale(imageA_half, imageB_half, min_scale)
        x_offset = 2*x_offset - 1
        y_offset = 2*y_offset - 1
 
    dimension_x = 3
    dimension_y = 3
    imageA_array = normalize(np.array(imageA))
    imageB_array = normalize(np.array(imageB))

    max_score = evaluate(imageA_array, imageB_array, x_offset, y_offset)
    best_x = x_offset
    best_y = y_offset

    for i in range(dimension_y):
        for j in range(dimension_x):
            score = evaluate(imageA_array, imageB_array, x_offset+j, y_offset+i)
            #print(str(x_offset+j)+","+str(y_offset+i)+":"+str(score))
            if(score > max_score):
                max_score = score
                best_x = x_offset+j
                best_y = y_offset+i
                #print(str(best_x)+","+str(best_y)+":"+str(score))

    if(show_progress):
        print("Current scale:"+str(imageA.width)+","+str(imageA.height))

    return best_x, best_y


def crop_images(imageA, imageB, imageC, offset_bx, offset_by, offset_cx, offset_cy):
    
    crop_ax = 0
    crop_ay = 0
    crop_aw = imageA.width
    crop_ah = imageA.height

    crop_bx = 0
    crop_by = 0
    crop_bw = imageB.width
    crop_bh = imageB.height

    crop_cx = 0
    crop_cy = 0
    crop_cw = imageC.width
    crop_ch = imageC.height

    if(offset_bx > crop_ax):
        crop_ax = offset_bx
    if(offset_cx > crop_ax):
        crop_ax = offset_cx
    if(offset_by > crop_ay):
        crop_ay = offset_by
    if(offset_cy > crop_ay):
        crop_ay = offset_cy
    
    if(offset_bx + imageB.width < crop_aw):
        crop_aw = offset_bx + imageB.width
    if(offset_cx + imageC.width < crop_aw):
        crop_aw = offset_cx + imageC.width
    if(offset_by + imageB.height < crop_ah):
        crop_ah = offset_by + imageB.height
    if(offset_cy + imageC.height < crop_ah):
        crop_ah = offset_cy + imageC.height

    if(crop_ax > offset_bx):
        crop_bx = crop_ax - offset_bx
    if(crop_ay > offset_by):
        crop_by = crop_ay - offset_by
    if(crop_aw < offset_bx + imageB.width):
        crop_bw = crop_aw - offset_bx
    if(crop_ah < offset_by + imageB.height):
        crop_bh = crop_ah - offset_by
    
    if(crop_ax > offset_cx):
        crop_cx = crop_ax - offset_cx
    if(crop_ay > offset_cy):
        crop_cy = crop_ay - offset_cy
    if(crop_aw < offset_cx + imageC.width):
        crop_cw = crop_aw - offset_cx
    if(crop_ah < offset_cy + imageC.height):
        crop_ch = crop_ah - offset_cy

    return imageA.crop((crop_ax, crop_ay, crop_aw, crop_ah)), imageB.crop((crop_bx, crop_by, crop_bw, crop_bh)), imageC.crop((crop_cx, crop_cy, crop_cw, crop_ch))

    
def main():
    parser = argparse.ArgumentParser(description='Given 3 images, one for each color channel, it will try to align them as closely as possible.\nsingle: Finds the best alignment within the search radius(default: 8x8) using the single pass method\nmulti: Aligns the images using the multi pass method by recursive downsampling down to a given minimum resolution(default 32x32)\nboth: Aligns the images using both methods so you can compare their runtimes.')
    parser.add_argument("mode", help="Which algorithm to use", choices=['single', 'multi', 'both'], type=str)
    parser.add_argument("image_1", help="Image 1 file path", type=str)
    parser.add_argument("image_2", help="Image 2 file path", type=str)
    parser.add_argument("image_3", help="Image 3 file path", type=str)
    parser.add_argument("-o","--output", help="Output image file path", type=str, default="")
    parser.add_argument("-r","--radius", help="Radius in pixels to search when using single pass method", nargs = 2, type=int, default= [8,8])
    parser.add_argument("-s","--min_scale", help="The minimum scale for recursive downsampling when using multi pass method", type=int, default = 32)
    parser.add_argument("-g2","--guess_2", help="Initial offset for image 2 when using single pass method", nargs = 2, type=int, default= [None,None])
    parser.add_argument("-g3","--guess_3", help="Initial offset for image 3 when using single pass method", nargs = 2, type=int, default= [None,None])
    parser.add_argument("-a","--all_combinations", help="Show all color channel combinations", action="store_true")
    parser.add_argument("-p","--show_progress", help="Display information about the progress while processing", action="store_true")
    parser.add_argument("-t","--time", help="Measure the runtime of the alignment", action="store_true")

    args = parser.parse_args()
    mode = args.mode

    imageA = Image.open(args.image_1) #Load the three images, throws error if file path is invalid.
    imageB = Image.open(args.image_2)
    imageC = Image.open(args.image_3)
    output_path = args.output

    radius_x = args.radius[0] #Search radius for single scale method.  Search size is 2*radius+1
    radius_y = args.radius[1]
    min_scale = args.min_scale #Minimum size to downsample for multiscale method. Either dimension cannot be smaller than this value. 

    all_combinations = args.all_combinations #Whether to display all 6 possible color channel combinations
    global show_progress
    show_progress = args.show_progress #Whether to show details about the current progress.
    show_time = args.time #Whether to measure runtime.  
    
    #if(len(sys.argv) < 5 or len(sys.argv) > 7):
        #print("Usage:")
        #print("a1.py single <Image 1 path> <Image 2 path> <Image 3 path> [search_radius]")
        #print("Finds the best alignment of the 3 images using the single pass method within the search radius(default 8)\n")
        
        #print("a1.py multi <Image 1 path> <Image 2 path> <Image 3 path> [min_scale]")
        #print("Aligns the 3 images using the multi pass method by recursive downsampling to a given minimum resolution(default 32x32)\n")
        
        #print("a1.py compare <Image 1 path> <Image 2 path> <Image 3 path> [search_radius] [min_scale]")
        #print("Aligns the 3 images using both methods and compares the runtimes\n")

        #print("Show all color channel combinations? // Only supports 8 bit formats")
    #sys.exit()


    if(mode == "single" or mode == "both"):

        print("Single pass method:")
        print("Aligning image 2 to 1")

        if(show_time):
            start_time = time.time()

        alignment_bx, alignment_by = align(imageA, imageB, radius_x, radius_y, show_time, args.guess_2[0], args.guess_2[1])
        print("Aligning image 3 to 1")
        alignment_cx, alignment_cy = align(imageA, imageC, radius_x, radius_y, show_time, args.guess_3[0], args.guess_3[1])

        if(show_time):
            end_time = time.time()
            single_time = end_time - start_time
            print("Single pass run time:"+str(single_time))

        imageA_crop, imageB_crop, imageC_crop = crop_images(imageA, imageB, imageC, alignment_bx, alignment_by, alignment_cx, alignment_cy)
        print("Image 2 offset:"+str(alignment_bx)+","+str(alignment_by))
        print("Image 3 offset:"+str(alignment_cx)+","+str(alignment_cy))

        if(all_combinations):
            Image.merge("RGB",(imageA_crop, imageB_crop, imageC_crop)).show()
            Image.merge("RGB",(imageC_crop, imageA_crop, imageB_crop)).show()
            Image.merge("RGB",(imageB_crop, imageC_crop, imageA_crop)).show()
            Image.merge("RGB",(imageA_crop, imageC_crop, imageB_crop)).show()
            Image.merge("RGB",(imageB_crop, imageA_crop, imageC_crop)).show()
        result = Image.merge("RGB",(imageC_crop, imageB_crop, imageA_crop))
        result.show()
        if(output_path != ""):
            print("Image saved")
            result.save(output_path)



    if(mode == "multi" or mode == "both"):
        print("Multiscale method:")
        print("Aligning image 2 to 1")

        if(show_time):
            start_time = time.time()

        alignment_bx, alignment_by = align_multiscale(imageA, imageB, min_scale)
        print("Aligning image 3 to 1")
        alignment_cx, alignment_cy = align_multiscale(imageA, imageC, min_scale)

        if(show_time):
            end_time = time.time()
            multi_time = end_time - start_time
            print("Multipass run time:"+str(multi_time))
        
        imageA_crop, imageB_crop, imageC_crop = crop_images(imageA, imageB, imageC, alignment_bx, alignment_by, alignment_cx, alignment_cy)
        print("Image 2 offset:"+str(alignment_bx)+","+str(alignment_by))
        print("Image 3 offset:"+str(alignment_cx)+","+str(alignment_cy))

        if(all_combinations):
            Image.merge("RGB",(imageA_crop, imageB_crop, imageC_crop)).show()
            Image.merge("RGB",(imageC_crop, imageA_crop, imageB_crop)).show()
            Image.merge("RGB",(imageB_crop, imageC_crop, imageA_crop)).show()
            Image.merge("RGB",(imageA_crop, imageC_crop, imageB_crop)).show()
            Image.merge("RGB",(imageB_crop, imageA_crop, imageC_crop)).show()
        result = Image.merge("RGB",(imageC_crop, imageB_crop, imageA_crop))
        result.show()
        if(output_path != ""):
            print("Image saved")
            result.save(output_path)
            


    #imageA_crop.show()
    #imageB_crop.show()
    #imageC_crop.show()
    #Image.merge("RGB",(imageA_crop, imageB_crop, imageC_crop)).show()



if __name__ == "__main__":
    main()

    
def crop_images2(imageA, imageB, x_offset, y_offset):
    imageA.size
    crop_ax = 0
    crop_ay = 0
    crop_aw = imageA.width
    crop_ah = imageA.height

    crop_bx = 0
    crop_by = 0
    crop_bw = imageB.width
    crop_bh = imageB.height

    if(x_offset>0):
        crop_ax = x_offset
    if(x_offset<0):
        crop_bx = -x_offset
    if(y_offset>0):
        crop_ay = y_offset
    if(y_offset<0):
        crop_by = -y_offset
    
    if(x_offset + imageB.width > imageA.width):
        crop_bw = imageA.width - x_offset 
    if(x_offset + imageB.width < imageA.width):
        crop_aw = x_offset + imageB.width
    
    if(y_offset + imageB.height > imageA.height):
        crop_bh = imageA.height - y_offset 
    if(y_offset + imageB.height < imageA.height):
        crop_ah = y_offset + imageB.height
    
    return imageA.crop((crop_ax, crop_ay, crop_aw, crop_ah)), imageB.crop((crop_bx, crop_by, crop_bw, crop_bh))


