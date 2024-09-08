python compress.py --compression_rate 0.33 \
    --target_tokens -1 \
    --force_tokens "+ -  * × / ÷ = \n ? . question: , [ ]" \

# python test_script.py

# python compress.py --compressor_name "phi2" \
#     --use_llmlingua2 False \
#     --target_tokens 200 \
#     --compression_rate 1 \
#     --force_tokens "+ -  * × / ÷ = \n ? . question: , [ ]"