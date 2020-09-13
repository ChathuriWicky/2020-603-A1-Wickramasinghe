/***
Author: Chathuri Wickrmasinghe, VCU, brahmanacsw@vcu.edu

Run using:
  make -f Makefile
  mpirun -np 4 ./main datasets/small.arff 5
***/

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <stdint.h>
#include <iterator>
#include<algorithm>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <bitset>
#include <time.h>
#include <map>
#include <vector>
#include <set>
#include<list>
#include<random>

#include <mpi.h>
using namespace std;

//This function calculate the mode value for a given array with given size. got from stackoverflow
int get_mode(int* class_array, int class_array_size) {

    int* ipRepetition = new int[class_array_size];
    for (int i = 0; i < class_array_size; ++i) {
        ipRepetition[i] = 0;
        int j = 0;
        bool bFound = false;
        while ((j < i) && (class_array[i] != class_array[j])) {
            if (class_array[i] != class_array[j]) {
                ++j;
            }
        }
        ++(ipRepetition[j]);
    }
    int index_max_repeated = 0;
    for (int i = 1; i < class_array_size; ++i) {
        if (ipRepetition[i] > ipRepetition[index_max_repeated]) {
            index_max_repeated = i;
        }
    }
    delete [] ipRepetition;
    return class_array[index_max_repeated];

}

//this function takes the distances, K and dataset, returns the mode class value for smallest K distances
int find_class(float* distances_temp, ArffData* dataset,  int K ){

  int no_of_data_records = dataset->num_instances();
  vector<float> distances(distances_temp, distances_temp + no_of_data_records);
    vector<int> index(distances.size(), 0);


    for (int i = 0 ; i != index.size() ; i++) {
        index[i] = i;
    }
    sort(index.begin(), index.end(),
        [&](const int& a, const int& b) {
            return (distances[a] < distances[b]);
        }
    );
    int* predictions = (int*)malloc(K * sizeof(int));

    for (int i = 0 ; i< K ; i++) {
        predictions[i] = dataset->get_instance(index[i])->get(dataset->num_attributes() - 1)->operator int32();
    }

    int predicted_class = get_mode(predictions, K) ;
    return predicted_class;
}

//this function calculates prediction for a given data range and for a given K value
int* KNN_openmp(ArffData* dataset, int K, int start_pos, int stop_pos)
{


  int no_of_datapoints = (stop_pos - start_pos) + 1;
  int* predictions = (int*)malloc(no_of_datapoints * sizeof(int));
  int temp=0;


  for(int i = start_pos; i < stop_pos+1; i++) // for each instance in the dataset
  {

      float smallestDistance = FLT_MAX;
      int smallestDistanceClass;
      float* distances = (float*)malloc(dataset->num_instances() * sizeof(float));
      distances[start_pos+ temp] = FLT_MAX;


      #pragma omp parallel for num_threads(56) //private(distance) shared(distances)
      for(int j = 0; j < dataset->num_instances(); j++) // target each other instance
      {
          if( start_pos+ temp == j) continue;

          float distance = 0;

          for(int k = 0; k < dataset->num_attributes() - 1; k++) // compute the distance between the two instances
          {
              float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
              distance += diff * diff;
          }

          distance = sqrt(distance);
          distances[j] = distance;

      }

      int predicated_class = find_class(distances, dataset, K);
      predictions[temp] = predicated_class;
      temp++;
      //std::cout << "For data point: " << i << " smallestDistanceClass: " << smallestDistanceClass << " Predicted with given K : " <<  temp_class << "\n";

  }

  return predictions;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses

    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;

    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / (float) dataset->num_instances();
}

//just for printing/testing
void print_elements(int* array, int size){
  std::cout << "Printing local elements" << "\n";
  for(int i=0;i<size;i++){
    std::cout << array[i] << " , ";
  }

}

//this function takes local predictions and fill the values into global prections array
void fill_main_predictions(int* predictions_main, int* predictions_local, int start_pos, int stop_pos){
  int temp=0;
  for(int i= start_pos; i<=stop_pos; i++){
    predictions_main[i] = predictions_local[temp];
    temp++;
  }

}

int main(int argc, char *argv[])
{

    if(argc != 3)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        cout << "Enter the Value for K " << endl;
        exit(0);
    }




    int size;
    int rank;
    MPI_Status status;
    const int VERY_LARGE_INT = 999999;
    const int ROOT = 0;
    int tag = 1234;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status Stat;


    //int no_of_processors = omp_get_num_procs(); //to get the available cpu cores (logical or physical, hyper-threading can play a role).

    //int no_of_threads = omp_get_max_threads();

    //std::cout << "no_of_processors: " << no_of_processors << ", no_of_threads" << no_of_threads << "\n";


    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    // Get the user input for K
    int K = atoi(argv[2]);

    struct timespec start, end;
    int no_of_datapoints = dataset->num_instances();
    int count = no_of_datapoints/(size);
    int remainder = no_of_datapoints % size;


    if (rank == ROOT) {

        std::cout << "Im root with rank " << rank <<  ", number of elements per processors is " << count << ", Total no of datapoints in the dataset is :" << no_of_datapoints << "\n";

        //if only 1 processor is given by the user
        if( size ==1) {

          int start_pos = 0;
          int stop_pos = count - 1;

          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          // Get the class predictions
          int* predictions = KNN_openmp(dataset, K, start_pos, stop_pos);
          //print_elements(predictions, no_of_datapoints);
          // Compute the confusion matrix
          int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
          // Calculate the accuracy
          float accuracy = computeAccuracy(confusionMatrix, dataset);

          clock_gettime(CLOCK_MONOTONIC_RAW, &end);
          uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

          printf("The KNN classifier with K = %lu, Serial Version for %lu instances required %llu ms CPU time, accuracy was %.4f\n", K, dataset->num_instances(), (long long unsigned int) diff, accuracy);

        } //if more than 1 processor is given by the user
        else{
            //timer starts
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);


            int start_pos = 0;
            int stop_pos = count - 1;
            std::cout << "Im the root, i have data points with index range: " << start_pos << " - " << stop_pos << "\n";

            //memory location for storing the results from all the processors
            int* predictions_main = (int*)malloc(no_of_datapoints * sizeof(int));

            // from root, calc local predictions for its data chunk with the given range
            int datapoints_count = (stop_pos - start_pos) + 1;
            int* local_predictions = KNN_openmp(dataset, K, start_pos, stop_pos);

            //add local predication to the main predictions array
            fill_main_predictions(predictions_main, local_predictions, start_pos, stop_pos);

            // get localprediction from other processors and store them in the main predictions array, Here last processor will get the remaining element if no of data points not devisable by the no of processors
            // it can be given to the root processor, but i dicided to put that extra load into the last processor
            for(int i = 1; i < size; i++){
                int start_pos, stop_pos;
                start_pos = i * (count);
                if ( i==size-1) {
                    stop_pos = start_pos + count- 1 + remainder;
                } else {
                    stop_pos = start_pos + (count - 1);
                }
                int no_of_ele = (stop_pos - start_pos) + 1;
                int *localArray = (int *)malloc(no_of_ele * sizeof(int));
                //receive local arrays
                MPI_Recv(localArray, no_of_ele, MPI_INT, i, 3, MPI_COMM_WORLD, &Stat);
                fill_main_predictions(predictions_main, localArray, start_pos, stop_pos);

            }

            int* confusionMatrix = computeConfusionMatrix(predictions_main, dataset);
            // Calculate the accuracy
            float accuracy = computeAccuracy(confusionMatrix, dataset);

            clock_gettime(CLOCK_MONOTONIC_RAW, &end);
            uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

            printf("The KNN classifier K = %lu,, Parallel Version for %lu instances required %llu ms CPU time, accuracy was %.4f\n", K , dataset->num_instances(), (long long unsigned int) diff, accuracy);



        }


    } // end of root

    else{

        int start_pos, stop_pos;
        if(rank!=size-1){
            start_pos = rank * count;
            stop_pos = rank * count + (count -1);
        }
        else{
          start_pos = rank * count;
          stop_pos = rank* count + (count -1)+ remainder;
        }
        int no_of_ele = (stop_pos - start_pos) + 1;
          std::cout << "Im rank " << rank << " , i have data point with the range: " << start_pos << " - " << stop_pos << " , No of data points to process is: " << no_of_ele << "\n";

        int* local_predictions = KNN_openmp(dataset, K, start_pos, stop_pos);
        //send local predications into the root
        MPI_Send(local_predictions, no_of_ele, MPI_INT, 0, 3, MPI_COMM_WORLD);


    } // end of other processors

    MPI_Finalize();
    return 0;

}
