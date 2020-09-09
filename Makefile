todo: main_v1 main_v2
main_v1: main_v1.cpp
	mpic++ -o main_v1 main_v1.cpp -std=gnu++11 -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
main_v2: main_v2.cpp
	mpic++ -o main_v2 main_v2.cpp -std=gnu++11 -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
clean:
	rm  main_v1 main_v2
