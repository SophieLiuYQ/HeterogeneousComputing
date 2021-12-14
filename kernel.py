analysis_kernel = """

__kernel void analyze_weights_1(__global int* words_by_letter, __global int* num_words_by_letter, volatile __global float* out_stats, int max_words_per_letter) {

    // Get the word for the current work-item to focus on

    unsigned int word_id = get_global_id(0);
    unsigned int letter_id = get_global_id(1);

    // Prepare the indices for the reduction

    unsigned int work_item_id = get_local_id(0);
    unsigned int work_group_x = get_group_id(0);
    unsigned int work_group_y = get_group_id(1);
    unsigned int group_size = get_local_size(0);

    // Create local arrays to store the data in
    // The value 512 must be equal to the group size. This is the only value in the kernel
        // - that must be updated when the group size is changed.

    volatile __local float local_out[6 * 512];

    // Get the weight and frequency for the current thread

    float weight = 0;   
    int frequency = 0;
    if (word_id < num_words_by_letter[letter_id]) {
        frequency = words_by_letter[letter_id * max_words_per_letter * 7 + word_id * 7 + 5];
        weight = (float)words_by_letter[letter_id * max_words_per_letter * 7 + word_id * 7 + 6] / frequency;
    }

    // Each thread loads initial data into its own space in local memory

    local_out[group_size * 0 + work_item_id] =  weight;
    local_out[group_size * 1 + work_item_id] =  frequency * weight;
    local_out[group_size * 2 + work_item_id] =  weight;
    local_out[group_size * 3 + work_item_id] =  (word_id < num_words_by_letter[letter_id]) ? weight : 1;
    local_out[group_size * 4 + work_item_id] =  (word_id < num_words_by_letter[letter_id]) ? 1 : 0;
    local_out[group_size * 5 + work_item_id] =  frequency;


    // Preform reduction

    for (unsigned int stride = 1; stride < group_size; stride *= 2) {

        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

        if ( work_item_id % stride == 0 && work_item_id < group_size / 2) {

            local_out[group_size * 0 + work_item_id * 2] +=  local_out[group_size * 0 + work_item_id * 2 + stride];
            local_out[group_size * 1 + work_item_id * 2] +=  local_out[group_size * 1 + work_item_id * 2 + stride];
            local_out[group_size * 2 + work_item_id * 2] =   local_out[group_size * 2 + work_item_id * 2 + stride] > local_out[group_size * 2 + work_item_id * 2] ? local_out[group_size * 2 + work_item_id * 2 + stride] : local_out[group_size * 2 + work_item_id * 2];
            local_out[group_size * 3 + work_item_id * 2] =   local_out[group_size * 3 + work_item_id * 2 + stride] < local_out[group_size * 3 + work_item_id * 2] ? local_out[group_size * 3 + work_item_id * 2 + stride] : local_out[group_size * 3 + work_item_id * 2];
            local_out[group_size * 4 + work_item_id * 2] +=  local_out[group_size * 4 + work_item_id * 2 + stride];
            local_out[group_size * 5 + work_item_id * 2] +=  local_out[group_size * 5 + work_item_id * 2 + stride];
        }
    }

    // Synchronize work items again to ensure all are done reduction before writeback

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    // Writeback to output

    if (work_item_id < 6) {
        out_stats[ (work_group_y * 5 + work_group_x) * 6 + work_item_id] = local_out[group_size * work_item_id];
    }
}

__kernel void analyze_weights_2(__global int* words_by_letter, __global int* num_words_by_letter, volatile __global float* out_stats, int max_words_per_letter, float average, float weighted_average) {

    // Get the word for the current work-item to focus on

    unsigned int word_id = get_global_id(0);
    unsigned int letter_id = get_global_id(1);

    // Prepare the indices for the reduction

    unsigned int work_item_id = get_local_id(0);
    unsigned int work_group_x = get_group_id(0);
    unsigned int work_group_y = get_group_id(1);
    unsigned int group_size = get_local_size(0);

    // Create local arrays to store the data in
    // The value 512 must be equal to the group size. This is the only value in the kernel
    // - that must be updated when the group size is changed.

    volatile __local float local_out[2 * 512];

    // Get the weight and frequency for the current thread

    float weight = 0;   
    int frequency = 0;
    if (word_id < num_words_by_letter[letter_id]) {
        frequency = words_by_letter[letter_id * max_words_per_letter * 7 + word_id * 7 + 5];
        weight = (float)words_by_letter[letter_id * max_words_per_letter * 7 + word_id * 7 + 6] / frequency;
    }

    // Each thread loads initial data into its own space in local memory - initialize the normal average to zero first to avoid incorrect initialization for out of bounds values

    local_out[group_size * 0 + work_item_id] = 0;

    if (word_id < num_words_by_letter[letter_id])
        local_out[group_size * 0 + work_item_id] =  (weight - average) * (weight - average);

    local_out[group_size * 1 + work_item_id] =  (weight - weighted_average) * (weight - weighted_average) * frequency;

    // Preform reduction

    for (unsigned int stride = 1; stride < group_size; stride *= 2) {

        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

        if ( work_item_id % stride == 0 && work_item_id < group_size / 2) {

            local_out[group_size * 0 + work_item_id * 2] +=  local_out[group_size * 0 + work_item_id * 2 + stride];
            local_out[group_size * 1 + work_item_id * 2] +=  local_out[group_size * 1 + work_item_id * 2 + stride];
        }
    }

    // Synchronize work items again to ensure all are done reduction before writeback

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    // Writeback to output

    if (work_item_id < 2) {
        out_stats[ (work_group_y * 5 + work_group_x) * 2 + work_item_id] = local_out[group_size * work_item_id];
    }
}

"""

predict_kernel = """

__kernel void predict(__global char* words, __global int* weights, __global char* weights_char, __global int* num_weights_letter, volatile __global float* out_weights, int max_words_per_letter, int word_max) {

    // Get the word for the current work-item to focus on

    unsigned int word_id = get_global_id(0);

    unsigned int word_index = word_id * 16;

    // Get the weight forthe current work item to focus on

    unsigned int weight_id = get_global_id(1);

    if ( word_id < word_max ) 
    {
        unsigned int letter_index;
        if (words[word_index] > 96) 
            letter_index = words[word_index] - 'a';
        else
            letter_index = words[word_index] - 'A';

        unsigned int weight_index = letter_index * max_words_per_letter * 28 + weight_id * 28;
        unsigned int weight_index_int = letter_index * max_words_per_letter * 7 + weight_id * 7;
        unsigned int weight_max = letter_index * max_words_per_letter * 28 + num_weights_letter[letter_index] * 28;

        // Get the inputs and outputs to be compared

        char word_0 = words[word_index + 0];
        char word_1 = words[word_index + 1];
        char word_2 = words[word_index + 2];
        char word_3 = words[word_index + 3];
        char word_4 = words[word_index + 4];
        char word_5 = words[word_index + 5];
        char word_6 = words[word_index + 6];
        char word_7 = words[word_index + 7];
        char word_8 = words[word_index + 8];
        char word_9 = words[word_index + 9];
        char word_10 = words[word_index + 10];
        char word_11 = words[word_index + 11];
        char word_12 = words[word_index + 12];
        char word_13 = words[word_index + 13];
        char word_14 = words[word_index + 14];
        char word_15 = words[word_index + 15];

        if ( weight_index < weight_max ) 
        {
            char word_w_0 = weights_char[weight_index + 0];
            char word_w_1 = weights_char[weight_index + 1];
            char word_w_2 = weights_char[weight_index + 2];
            char word_w_3 = weights_char[weight_index + 3];
            char word_w_4 = weights_char[weight_index + 4];
            char word_w_5 = weights_char[weight_index + 5];
            char word_w_6 = weights_char[weight_index + 6];
            char word_w_7 = weights_char[weight_index + 7];
            char word_w_8 = weights_char[weight_index + 8];
            char word_w_9 = weights_char[weight_index + 9];
            char word_w_10 = weights_char[weight_index + 10];
            char word_w_11 = weights_char[weight_index + 11];
            char word_w_12 = weights_char[weight_index + 12];
            char word_w_13 = weights_char[weight_index + 13];
            char word_w_14 = weights_char[weight_index + 14];
            char word_w_15 = weights_char[weight_index + 15];

            // Compare them and update the output if necessary

            if ( word_0 == word_w_0 && word_1 == word_w_1 && word_2 == word_w_2 && word_3 == word_w_3 &&
                 word_4 == word_w_4 && word_5 == word_w_5 && word_6 == word_w_6 && word_7 == word_w_7 &&
                 word_8 == word_w_8 && word_9 == word_w_9 && word_10 == word_w_10 && word_11 == word_w_11 &&
                 word_12 == word_w_12 && word_13 == word_w_13 && word_14 == word_w_14 && word_15 == word_w_15)
            {
                int frequency = weights[weight_index_int + 5];
                float weight = (float)weights[weight_index_int + 6] / frequency;

                out_weights[word_id] = weight;
            }
        }
    }
}

__kernel void predict_bayes(__global char* words, __global int* weights, __global char* weights_char, __global int* num_weights_letter, volatile __global float* out_probabilities_up, volatile __global float* out_probabilities_down, int total_weights_up, int total_weights_down, int total_weights, float c, int max_words_per_letter, int word_max) {

    // Get the word for the current work-item to focus on

    unsigned int word_id = get_global_id(0);

    unsigned int word_index = word_id * 16;

    // Get the weight forthe current work item to focus on

    unsigned int weight_id = get_global_id(1);

    if ( word_id < word_max ) 
    {
        unsigned int letter_index;
        if (words[word_index] > 96) 
            letter_index = words[word_index] - 'a';
        else
            letter_index = words[word_index] - 'A';

        unsigned int weight_index = letter_index * max_words_per_letter * 28 + weight_id * 28;
        unsigned int weight_index_int = letter_index * max_words_per_letter * 7 + weight_id * 7;
        unsigned int weight_max = letter_index * max_words_per_letter * 28 + num_weights_letter[letter_index] * 28;

        // Get the inputs and outputs to be compared

        char word_0 = words[word_index + 0];
        char word_1 = words[word_index + 1];
        char word_2 = words[word_index + 2];
        char word_3 = words[word_index + 3];
        char word_4 = words[word_index + 4];
        char word_5 = words[word_index + 5];
        char word_6 = words[word_index + 6];
        char word_7 = words[word_index + 7];
        char word_8 = words[word_index + 8];
        char word_9 = words[word_index + 9];
        char word_10 = words[word_index + 10];
        char word_11 = words[word_index + 11];
        char word_12 = words[word_index + 12];
        char word_13 = words[word_index + 13];
        char word_14 = words[word_index + 14];
        char word_15 = words[word_index + 15];

        if ( weight_index < weight_max ) 
        {
            char word_w_0 = weights_char[weight_index + 0];
            char word_w_1 = weights_char[weight_index + 1];
            char word_w_2 = weights_char[weight_index + 2];
            char word_w_3 = weights_char[weight_index + 3];
            char word_w_4 = weights_char[weight_index + 4];
            char word_w_5 = weights_char[weight_index + 5];
            char word_w_6 = weights_char[weight_index + 6];
            char word_w_7 = weights_char[weight_index + 7];
            char word_w_8 = weights_char[weight_index + 8];
            char word_w_9 = weights_char[weight_index + 9];
            char word_w_10 = weights_char[weight_index + 10];
            char word_w_11 = weights_char[weight_index + 11];
            char word_w_12 = weights_char[weight_index + 12];
            char word_w_13 = weights_char[weight_index + 13];
            char word_w_14 = weights_char[weight_index + 14];
            char word_w_15 = weights_char[weight_index + 15];

            // Compare them and update the output if necessary

            if ( word_0 == word_w_0 && word_1 == word_w_1 && word_2 == word_w_2 && word_3 == word_w_3 &&
                 word_4 == word_w_4 && word_5 == word_w_5 && word_6 == word_w_6 && word_7 == word_w_7 &&
                 word_8 == word_w_8 && word_9 == word_w_9 && word_10 == word_w_10 && word_11 == word_w_11 &&
                 word_12 == word_w_12 && word_13 == word_w_13 && word_14 == word_w_14 && word_15 == word_w_15)
            {
                float up = ((float)weights[weight_index_int + 5] + c) / (total_weights_up + c * total_weights);
                float down = ((float)weights[weight_index_int + 6] + c) / (total_weights_down + c * total_weights);

                out_probabilities_up[word_id] = up;
                out_probabilities_down[word_id] = down;
            }
        }
    }
}
"""

update_kernel = """

__kernel void update(__global char* words, volatile __global int* word_bitmap, volatile __global int* weights, __global char* weights_char, __global int* num_weights_letter, int max_words_per_letter, int word_max, int direction) {

    // Get the word for the current work-item to focus on

    unsigned int word_id = get_global_id(0);

    unsigned int word_index = word_id * 16;

    // Get the weight for the current work item to focus on

    unsigned int weight_id = get_global_id(1);

    if ( word_id < word_max ) 
    {
        unsigned int letter_index;
        if (words[word_index] > 96) 
            letter_index = words[word_index] - 'a';
        else
            letter_index = words[word_index] - 'A';

        unsigned int weight_index = letter_index * max_words_per_letter * 28 + weight_id * 28;
        unsigned int weight_index_int = letter_index * max_words_per_letter * 7 + weight_id * 7;
        unsigned int weight_max = letter_index * max_words_per_letter * 28 + num_weights_letter[letter_index] * 28;

        // Get the inputs and outputs to be compared

        char word_0 = words[word_index + 0];
        char word_1 = words[word_index + 1];
        char word_2 = words[word_index + 2];
        char word_3 = words[word_index + 3];
        char word_4 = words[word_index + 4];
        char word_5 = words[word_index + 5];
        char word_6 = words[word_index + 6];
        char word_7 = words[word_index + 7];
        char word_8 = words[word_index + 8];
        char word_9 = words[word_index + 9];
        char word_10 = words[word_index + 10];
        char word_11 = words[word_index + 11];
        char word_12 = words[word_index + 12];
        char word_13 = words[word_index + 13];
        char word_14 = words[word_index + 14];
        char word_15 = words[word_index + 15];

        if ( weight_index < weight_max ) 
        {
            char word_w_0 = weights_char[weight_index + 0];
            char word_w_1 = weights_char[weight_index + 1];
            char word_w_2 = weights_char[weight_index + 2];
            char word_w_3 = weights_char[weight_index + 3];
            char word_w_4 = weights_char[weight_index + 4];
            char word_w_5 = weights_char[weight_index + 5];
            char word_w_6 = weights_char[weight_index + 6];
            char word_w_7 = weights_char[weight_index + 7];
            char word_w_8 = weights_char[weight_index + 8];
            char word_w_9 = weights_char[weight_index + 9];
            char word_w_10 = weights_char[weight_index + 10];
            char word_w_11 = weights_char[weight_index + 11];
            char word_w_12 = weights_char[weight_index + 12];
            char word_w_13 = weights_char[weight_index + 13];
            char word_w_14 = weights_char[weight_index + 14];
            char word_w_15 = weights_char[weight_index + 15];

            // Compare them and update the output if necessary

            if ( word_0 == word_w_0 && word_1 == word_w_1 && word_2 == word_w_2 && word_3 == word_w_3 &&
                 word_4 == word_w_4 && word_5 == word_w_5 && word_6 == word_w_6 && word_7 == word_w_7 &&
                 word_8 == word_w_8 && word_9 == word_w_9 && word_10 == word_w_10 && word_11 == word_w_11 &&
                 word_12 == word_w_12 && word_13 == word_w_13 && word_14 == word_w_14 && word_15 == word_w_15)
            {
                if (direction == 1)
                {
                    atomic_inc(weights + weight_index_int + 5);
                    atomic_inc(weights + weight_index_int + 6);
                }
                else
                {
                    atomic_inc(weights + weight_index_int + 5);
                }

                word_bitmap[word_id] = 1;
            }
        }
    }
}

__kernel void update_bayes(__global char* words, volatile __global int* word_bitmap, volatile __global int* weights, __global char* weights_char, __global int* num_weights_letter, int max_words_per_letter, int word_max, int direction) {

    // Get the word for the current work-item to focus on

    unsigned int word_id = get_global_id(0);

    unsigned int word_index = word_id * 16;

    // Get the weight for the current work item to focus on

    unsigned int weight_id = get_global_id(1);

    if ( word_id < word_max ) 
    {
        unsigned int letter_index;
        if (words[word_index] > 96) 
            letter_index = words[word_index] - 'a';
        else
            letter_index = words[word_index] - 'A';

        unsigned int weight_index = letter_index * max_words_per_letter * 28 + weight_id * 28;
        unsigned int weight_index_int = letter_index * max_words_per_letter * 7 + weight_id * 7;
        unsigned int weight_max = letter_index * max_words_per_letter * 28 + num_weights_letter[letter_index] * 28;

        // Get the inputs and outputs to be compared

        char word_0 = words[word_index + 0];
        char word_1 = words[word_index + 1];
        char word_2 = words[word_index + 2];
        char word_3 = words[word_index + 3];
        char word_4 = words[word_index + 4];
        char word_5 = words[word_index + 5];
        char word_6 = words[word_index + 6];
        char word_7 = words[word_index + 7];
        char word_8 = words[word_index + 8];
        char word_9 = words[word_index + 9];
        char word_10 = words[word_index + 10];
        char word_11 = words[word_index + 11];
        char word_12 = words[word_index + 12];
        char word_13 = words[word_index + 13];
        char word_14 = words[word_index + 14];
        char word_15 = words[word_index + 15];

        if ( weight_index < weight_max ) 
        {
            char word_w_0 = weights_char[weight_index + 0];
            char word_w_1 = weights_char[weight_index + 1];
            char word_w_2 = weights_char[weight_index + 2];
            char word_w_3 = weights_char[weight_index + 3];
            char word_w_4 = weights_char[weight_index + 4];
            char word_w_5 = weights_char[weight_index + 5];
            char word_w_6 = weights_char[weight_index + 6];
            char word_w_7 = weights_char[weight_index + 7];
            char word_w_8 = weights_char[weight_index + 8];
            char word_w_9 = weights_char[weight_index + 9];
            char word_w_10 = weights_char[weight_index + 10];
            char word_w_11 = weights_char[weight_index + 11];
            char word_w_12 = weights_char[weight_index + 12];
            char word_w_13 = weights_char[weight_index + 13];
            char word_w_14 = weights_char[weight_index + 14];
            char word_w_15 = weights_char[weight_index + 15];

            // Compare them and update the output if necessary

            if ( word_0 == word_w_0 && word_1 == word_w_1 && word_2 == word_w_2 && word_3 == word_w_3 &&
                 word_4 == word_w_4 && word_5 == word_w_5 && word_6 == word_w_6 && word_7 == word_w_7 &&
                 word_8 == word_w_8 && word_9 == word_w_9 && word_10 == word_w_10 && word_11 == word_w_11 &&
                 word_12 == word_w_12 && word_13 == word_w_13 && word_14 == word_w_14 && word_15 == word_w_15)
            {
                if (direction == 1)
                {
                    atomic_inc(weights + weight_index_int + 5);
                }
                else
                {
                    atomic_inc(weights + weight_index_int + 6);
                }

                word_bitmap[word_id] = 1;
            }
        }
    }
}
"""