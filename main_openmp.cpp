/***
Author: Chathuri Wickrmasinghe, VCU, brahmanacsw@vcu.edu

Run using:
  make -f Makefile
  ./main_openmp datasets/small.arff 5 56
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
int* KNNOpenMP(ArffData* dataset, int K, int no_of_threads)
{

  int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));

  #pragma omp parallel for num_threads(no_of_threads)
  for(int i = 0; i < dataset->num_instances() ; i++) // for each instance in the dataset
  {

      float smallestDistance = FLT_MAX;
      int smallestDistanceClass;
      float* distances = (float*)malloc(dataset->num_instances() * sizeof(float));
      distances[i] = FLT_MAX;

      //#pragma omp parallel for
      for(int j = 0; j < dataset->num_instances(); j++) // target each other instance
      {
          if(i == j) continue;

          float distance = 0;

          for(int k = 0; k < dataset->num_attributes() - 1; k++) // compute the distance between the two instances
          {
              float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
              distance += diff * diff;
          }

          distance = sqrt(distance);
          distances[j] = distance;

          //printf("\n data point index= %lu distance = %f class= %lu,",j,distance, dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32());
      }

      int predicted_class = find_class(distances, dataset, K);
      predictions[i] = predicted_class;
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

    if(argc != 4)
    {
        cout << "Usage: ./main datasets/datasetFile.arff K no_of_threads" << endl;

        exit(0);
    }




    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    // Get the user input for K
    int K = atoi(argv[2]);
    int no_of_threads = atoi(argv[3]);

    struct timespec start, end;
    uint64_t diff;


    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // Get the class predictions
    int* predictions = KNNOpenMP(dataset, K, no_of_threads);
    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, dataset);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    confusionMatrix = computeConfusionMatrix(predictions, dataset);
    accuracy = computeAccuracy(confusionMatrix, dataset);

    printf("The KNN classifier with K=%lu OpenMP for %lu instances required %llu ms CPU time, accuracy was %.4f\n", K, dataset->num_instances(), (long long unsigned int) diff, accuracy);

    return 0;

}
