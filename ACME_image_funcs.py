"""
This file contains a function called <read_hobj> which takes the full path to HOBJ binary image file,
reads the image data and returns the data as a numpy array formated into a raw monochromatic 12-bit interger 2D image.
The numpy array could then be post processed however the user chooses.
This functions replicates read_hobj fucntion used in OMA2 (see https://github.com/mblong).
This function can be used to read color camera image data from 
the Advanced Combustion Via Microgravity experiments (ACME). 

Created by Jesse A. Tinajero Jr. 
Coflow Laminar Diffusion (CLD) Flame
at Yale University
Created on 06/12/2020 
Note: This function has only been tested with python3.6.9 and python3.7.7

Example:
import ACME_image_funcs as aif
import numpy as np
import colour_demosaicing

def main():
	# READ THE IMAGE
	image_data =  aif.read_hobj('20180914_234125.024_ACME_18257J1_00000.hobj')

	# TO CONVERT RAW IMAGE DATA TO COLOR IMAGE RGB() IT IS POSSIBLE TO USE THE 
	# COLOUR_DEMOSAICING MODULE.
	image_data_demo =  colour_demosaicing.demosaicing_CFA_Bayer_bilinear(image_data, pattern='RGGB')

if __name__ == "__main__":
	main()
"""

import struct, time, os
import numpy as np

def read_hobj(image_filename):
	start_time = time.time()
	print('Getting HOBJ file >>> {:s}'.format(image_filename))
	with open(image_filename,'rb') as f:
		ibuffer 	= f.read(84)
		#
		npixels 	= struct.unpack('i',ibuffer[72:76][::-1])[0]
		rows 		= struct.unpack('h',ibuffer[80:82][::-1])[0]+1
		cols 		= struct.unpack('h',ibuffer[82:84][::-1])[0]+1
		#
		print('Image width/height is {:d}/{:d}.'.format(cols, rows))
		print('# of pixels is {:d}.'.format(npixels))
		#
		f.seek(84+rows*6+17)
		image_str 	= f.read(npixels*2)
		bit_format 	= '<{:d}{:s}'.format(npixels, 'h')
		image_array 	= struct.unpack(bit_format, image_str)
		print('Read time: {:.2f} seconds.'.format(time.time() - start_time))
	return np.array(image_array).reshape(rows , cols)





def read_oma( image_filename ):
	start_time 	= time.time()
	image_filename 	= image_filename.replace('.hobj','')	
	print( 'Getting oma file >>> {:s}'.format(image_filename))
	image_file	 	= open( image_filename ,'rb')
	image_header	 	= image_file.read(30)
	if b'OMA2 Binary Data' in image_header:
		nspecs, nvalues, nrulerchar	= struct.unpack('iii' , image_file.read(3*4) )
		#
		specs 		= struct.unpack('{:d}{:s}'.format(nspecs, 'i') , image_file.read(4*nspecs) )
		#
		values 		= struct.unpack('{:d}{:s}'.format(nvalues, 'f') , image_file.read(4*nvalues) )
		#
		rulerchar 	= struct.unpack('{:d}{:s}'.format(nrulerchar, 'c') , image_file.read(1*nrulerchar) )
		#
		error, is_big_endian, commentSize, extraSize \
				= struct.unpack('iiii' , image_file.read(4*4) )
		#
		if commentSize>0:
			comment = struct.unpack('{:d}{:s}'.format(commentSize, 'c') , image_file.read(1*commentSize) )
		#
		if extraSize>0:
			extra 	= struct.unpack('{:d}{:s}'.format(extraSize, 'c') , image_file.read(1*extraSize) )
		#
		row 		= specs[0]
		col 		= specs[1]
		is_color 	= specs[8]
	else:
		image_file.seek(7*2)
		col, row 	= struct.unpack('hh' , image_file.read(4))
		image_file.seek(256*2+80*4)
		is_color	= 0
	npixels = row*col
	image_str 	= image_file.read(npixels*4)
	bit_format 	= '{:d}{:s}'.format(npixels, 'f')
	image 		= struct.unpack(bit_format, image_str)
	image 		= np.array(image).reshape(row , col)
	image_file.close()
	if is_color==1:
		row_size = int(row/3)
		red 			= image[0:row_size , :]
		green 			= image[row_size:2*row_size,:]
		blue			= image[2*row_size:3*row_size,:]
		color_image		= np.zeros((row_size,col,3))
		color_image[:,:,0] 	= red
		color_image[:,:,1] 	= green
		color_image[:,:,2] 	= blue
		print('{:f} seconds'.format(time.time()-start_time))
		return color_image
	else:
		print('{:f} seconds'.format(time.time()-start_time))
		return image

def write_oma( data , oma_filename, use_new=True ):
	start_time = time.time()
	if '.o2d' not in oma_filename: 
		oma_filename	= '{:s}.o2d'.format(oma_filename)
	print( 'Writing oma file >>> {:s}'.format(oma_filename))
	oma_file	 	= open( oma_filename ,'wb' )
	if use_new:
		header			= b'OMA2 Binary Data 1.0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
		oma_file.write( header )
		# NSPECS=32, NVALUE=16, NRULERCHAR=16
		oma_file.write( struct.pack('iii',32,16,16 ) )		

		# WRITE OUT SPECS. SPECS[0]=ROWS, SPECS[1]=COLS
		row = data.shape[0]
		if len(data.shape)>1:
			col		= data.shape[1]
		else:
			col 		= 1
		if len(data.shape)>2 and data.shape[2]==3:
			is_color 	= 1	
			data 		= np.vstack([ data[:,:,0], data[:,:,1], data[:,:,2] ])
			row		= 3*row
		else:
			is_color 	= 0	
		specs = np.array([row, col, 0, 0, 1, 1, 117315, 600, 1, 1, 0, 117315, 600, 117314, 26, 121200, 33, 0, 2, 0, 1, 0, 2, 0, 16, 0, 0, 0, 0, 0, 2791529, 0])
		specs[8] 		= is_color 
		for spec in specs:
			oma_file.write( struct.pack('i' , spec) )		
		#
		for value in range(16):
			oma_file.write( struct.pack('f' , 0.0) )
		#
		for ruler in range(16):
			oma_file.write( struct.pack('c' , os.urandom(1)) )

		# WRITE ERROR=0, IS_BIG_ENDIAN=0, COMMENTSIZE=0, EXTRASIZE=0
		oma_file.write( struct.pack('iiii', 0,0,0,0 ) )	
		# WRITE DATA
		data_single_row 		= np.reshape( data , row*col )
		for d in data_single_row:
			data_bytes		= struct.pack('f' , d) 	
			oma_file.write( data_bytes )		
	else:
		# THIS MEANS LITTLE_ENDIAN
		le 			= 32639
		# THIS IS BIG_ENDIAN
		be 			= 0
		header 			= np.zeros(256,'uint16')
		data_offset 		= np.zeros(80)
		header[226] 		= le
		row 			= data.shape[0]
		if len(data.shape)>1:
			col		= data.shape[1]
		else:
			col 		= 1
		header[7] 		= col
		header[8] 		= row
		for h in header:
			header_bytes		= struct.pack('H' , h) 	
			oma_file.write( header_bytes )		
		for o in data_offset:
			data_offset_bytes	= struct.pack('f' , o) 	
			oma_file.write( data_offset_bytes )		
		if len(data.shape)==3:
			data 	= np.vstack([ data[:,:,0], data[:,:,1], data[:,:,2] ])
			row	= 3*row
		data_single_row 		= np.reshape( data , row*col )
		for d in data_single_row:
			data_bytes		= struct.pack('f' , d) 	
			oma_file.write( data_bytes )		
	oma_file.close()		
	print('{:f} seconds'.format(time.time()-start_time))
