This README.txt file was generated on 20210628 by Brady Moon

-------------------
GENERAL INFORMATION
-------------------

1. Title of Dataset: TrajAir: A General Aviation Trajectory Dataset


2. Author Information

Author Contact Information
    Name: Jay Patrikar
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, Pennsylvania, USA, 15213
    Email: jpatrika@andrew.cmu.edu
	Phone Number: 

Author Contact Information
    Name: Brady Moon
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, Pennsylvania, USA, 15213
    Email: bradym@andrew.cmu.edu
	Phone Number: 

Author Contact Information 
    Name: Sourish Ghosh
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, Pennsylvania, USA, 15213
    Email: sourishg@andrew.cmu.edu
	Phone Number: 

Author Contact Information
    Name: Jean Oh
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, Pennsylvania, USA, 15213
    Email: jeanoh@cmu.edu
	Phone Number: 

Author Contact Information
    Name: Sebastian Scherer
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, Pennsylvania, USA, 15213
    Email: basti@andrew.cmu.edu
	Phone Number: 

---------------------
DATA & FILE OVERVIEW
---------------------

Directory of Files:
   A. Filename: 7day1     
      Short description: This folder contains seven contiguous days of processed and raw trajectory files.    


        
   B. Filename: 7day2     
      Short description: This folder contains seven contiguous days of processed and raw trajectory files.        


        
   C. Filename: 7day3     
      Short description: This folder contains seven contiguous days of processed and raw trajectory files. 



   D. Filename: 7day4 
      Short description: This folder contains seven contiguous days of processed and raw trajectory files. 



   E. Filename: 111_days    
      Short description: Contains the entire 111 days of processed and raw trajectory files.



   E. Filename: weather_data    
      Short description: Contains a csv file of the weather during the data collection period.
        

File Naming Convention: Processed files are numbered sequentially as <scene_num>.txt, whereas raw data folders are named as MM-DD-YY_adsb.


-----------------------------------------
DATA DESCRIPTION FOR: processed_data txt files
-----------------------------------------


1. Number of variables: 7


2. Number of cases/rows: 2731256


3. Missing data codes:


4. Variable List
    Column 1 = Frame Number
    Column 2 = Aircraft ID
    Column 3 = x (kilometers)
    Column 4 = y (kilometers)
    Column 5 = z (kilometers)
    Column 6 = wind_x (meters/second)
    Column 7 = wind_y (meters/second)




    A. Name: Frame Number
       Description: Frames are at a rate of 1 Hz. 


    B. Name: Aircraft ID
       Description: As recorded by the ADS-B


    C. Name: x (kilometers)
       Description: x position in airport centered inertial frame, with origin centered at end of runway and x axis aligned with runway.


    D. Name: y (kilometers)
       Description: y position in airport centered inertial frame, with origin centered at end of runway and x axis aligned with runway.



    E. Name: z (kilometers)
       Description: z position in airport centered inertial frame, with origin centered at end of runway and x axis aligned with runway.



    F. Name: wind_x (meters/second)
       Description: Wind speed in the x direction in meters per second


    G. Name: wind_y
       Description: Wind speed in the y direction in meters per second

--------------------------
METHODOLOGICAL INFORMATION
--------------------------


1. Equipment-specific information:

FlightBox ADS-B
Manufacturer: Open Flight Solutions
Link: https://www.openflightsolutions.com/flightbox/


3. Date of data collection range <YYYYMMDD>: 20200912-20210427

