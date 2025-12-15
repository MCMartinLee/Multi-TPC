# This script measures pitch, intensity and formants every 10ms
# Written by Setsuko Shirai
# Contact ssetsuko@u.washington.edu
# ask a user the directories
form supply_arguments
	sentence input_directory  A:
	sentence output_directory A:
	sentence type_file wav
	positive prediction_order 10
	positive minimum_pitch 75
	positive maximum_pitch 600
	positive new_sample_rate 11025
endform

# finding files we are looking for
Create Strings as file list... list 'input_directory$'\*.'type_file$'
# the name of files - later we could track each file
numberOfFiles = Get number of strings
for ifile to numberOfFiles
	select Strings list
	fileName$ = Get string... ifile
	Read from file... 'input_directory$'\'fileName$'
endfor

select all
numSelected = numberOfSelected ("Sound")

# change the name of each file - for batch processing
for i to numSelected
	select all
	currSoundID = selected ("Sound", i)
	select 'currSoundID'
	currName$ = "word_'i'"
	Rename... 'currName$'
endfor

for i to numSelected 
	select Sound word_'i'
	# get the finishing time of the Sound file
	fTime = Get finishing time
      	# Use numTimes in the loop
	numTimes = fTime / 0.01
	newName$ = "word_'i'"
	select Sound word_'i'
      	# 1st argument: New sample rate 2nd argument: Precision (samples)
	Resample... 'new_sample_rate' 50
	# 1st argument: Time step (s), 2nd argument: Minimum pitch for Analysis, 
	# 3rd argument: Maximum pitch for Analysis
	To Pitch... 0.01 'minimum_pitch' 'maximum_pitch'
	Rename... 'newName$'

	select Sound word_'i'_'new_sample_rate'
	To Intensity... 100 0
	
	select Sound word_'i'_'new_sample_rate'
	# 1st argument:  prediction order, 2nd argument: Analysis width (seconds)
	# 3rd argument: Time step (seconds),  4th argument: Pre-emphasis from (Hz)
	To LPC (autocorrelation)... prediction_order  0.025 0.005 50
	To Formant


	Create Table... table_word_'i' numTimes 3
	Set column label (index)... 1 time
	Set column label (index)... 2 pitch
	Set column label (index)... 3 intensity


	for itime to numTimes
		select Pitch word_'i'
		curtime = 0.01 * itime
		f0 = 0
		f0 = Get value at time... 'curtime' Hertz Linear
		f0$ = fixed$ (f0, 2)

		if f0$ = "--undefined--"
			f0$ = "0"
		endif


		curtime$ = fixed$ (curtime, 5)
		 
		select Intensity word_'i'_'new_sample_rate'
		intensity = Get value at time... 'curtime' Cubic
		
		
		intensity$ = fixed$ (intensity, 2)
		if intensity$ = "--undefined--"
			intensity$ = "0"
		endif



		select Table table_word_'i'
		Set numeric value... itime time 'curtime$'
		Set numeric value... itime pitch 'f0$' 
		Set numeric value... itime intensity  'intensity$'

	endfor
	select Strings list
	fileName$ = Get string... i
	select Table table_word_'i'
	Write to table file... 'output_directory$'\01-28-2022_'fileName$'.xls
endfor
