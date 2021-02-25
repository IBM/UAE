1. Title: Smartphone Dataset for Human Activity Recognition (HAR) in Ambient Assisted Living (AAL)

2. Relevant Information:
   -- This dataset is an addition to the dataset at 
      https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
      We collected more dataset to improve the accuracy of our HAR algorithms applied in 
      a Social connectedness experiment in the domain of Ambient Assisted Living. 
      The dataset was collected from the in-built accelerometer and gyroscope of a 
      smartphone worn around the waist of participants. See waist_mounted_phone.PNG.
      The data was collected from 30 participants within the age group of 22-79 years. 
      Each activity (standing, sitting, laying, walking, walking upstairs, walking downstairs) was 
      performed for 60secs and the 3-axial linear acceleration and 3-axial angular velocity  were 
      collected at a constant rate of 50Hz.
   -- These results presented in a paper titled: 'Activity Recognition Based on Inertial Sensors for Ambient Assisted Living' has been 
       accepted for publication in 19th International Conference on Information Fusion.

2. Source Information
   -- Creators: Kadian Alicia Davis (1), Evans Boateng Owusu (2)
     1 -- Department of Electrical, Electronic, Telecommunications Engineering and Naval Architecture (DITEN), 
          University of Genova, Genoa - Italy
     2 -- Independent Researcher, 
          Eindhoven, 
          The Netherlands
      Donors: E. B. Owusu (owboateng@gmail.com), K. A. Davis (kadian.davis@gmail.com)  
   -- Date: March, 2016
 
3. Past Usage:
    1. This dataset together with the dataset at:
     https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
     was used in a social connectedness experiment.
       --Results: We obtained an overall accuracy of 97.6% with Multiclass SVM for HAR,
                  91.4% for Hybrid SVM+HMM and 99.7% for Artificial Neural Networks (ANNs).
                  The accuracies were obatined through a 10-fold cross-validation.

5. Number of Instances: 5744 

6. Number of Attributes: 561 (See features.txt and features_info.txt included, taken from 
   https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) 

7. Attribute Information:
   For each record in the dataset it is provided: 
   - Triaxial acceleration from the accelerometer (total acceleration). 
     Filenames: final_acc_train.txt, final_acc_test.txt
   - Triaxial Angular velocity from the gyroscope. 
     Filenames: final_gyro_train.txt, final_gyro_test.txt
   - A 561-feature vector with time and frequency domain variables 
     (extracted from the triaxial data) 
     Filenames: final_X_train.txt, final_X_test.txt
     For more information about the features extracted see (features.txt and features_info.txt)
   - The corresponding activity labels. Filenames: final_y_train.txt and final_y_test.txt
 
