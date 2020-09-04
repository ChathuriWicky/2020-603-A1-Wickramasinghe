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

using namespace std;

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
    int iMaxRepeat = 0;
    for (int i = 1; i < class_array_size; ++i) {
        if (ipRepetition[i] > ipRepetition[iMaxRepeat]) {
            iMaxRepeat = i;
        }
    }
    delete [] ipRepetition;
    return class_array[iMaxRepeat];

}

int find_class(float* distances_temp, ArffData* dataset,  int K ){
  //cout << "in the function----------------------------" << "\n";

  int no_of_data_records = dataset->num_instances();

  //std::cout << "no_of_data_records: " << no_of_data_records <<" \n";

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
        //cout << index[i] << endl;
        predictions[i] = dataset->get_instance(index[i])->get(dataset->num_attributes() - 1)->operator int32();
        //std::cout << " predictions[i]  " << predictions[i]  << "\n";
    }

    int predicted_class = get_mode(predictions, K) ;
    //std::cout << "smallestDistance in the function" <<  distances_temp[index[0]]<< "\n Function end \n";
    //std::cout << "predicted_class" << predicted_class << "\n";
    return predicted_class;
}

int* KNN(ArffData* dataset, int K)
{
  cout << "K :" << K << "\n";
  int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));


  for(int i = 0; i < dataset->num_instances(); i++) // for each instance in the dataset
  {

      float smallestDistance = FLT_MAX;
      int smallestDistanceClass;
      float* distances = (float*)malloc(dataset->num_instances() * sizeof(float));
      distances[i] = FLT_MAX;

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
          if(distance < smallestDistance) // select the closest one
          {
              smallestDistance = distance;
              smallestDistanceClass = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32();
          }
      }




      int temp_class = find_class(distances, dataset, K);
      //std::cout << "closest class from the function" << dataset->get_instance(temp_class)->get(dataset->num_attributes() - 1)->operator int32() << "\n";
      //std::cout << "smallestDistanceClass from KNN" << smallestDistanceClass << '\n';
      predictions[i] = temp_class; //smallestDistanceClass;
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

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }




    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    int K=1;

    // Get the class predictions
    int* predictions = KNN(dataset, 5);
    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, dataset);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}
