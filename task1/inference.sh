docker load -i junqiangmler_sts24_task1_20240922_220244.tar.gz
# Do not change any of the parameters to docker run, these are fixed
# This is to mimic a restricted Grand-Challenge running environment
# ie no internet and no new privileges etc.
docker run --rm \
        -v F:/MedicalData/2024Semi-TeethSeg/docker_test/task1/inputs:/inputs \
        -v F:/MedicalData/2024Semi-TeethSeg/docker_test/task1/outputs:/outputs \
        --gpus=all \
        junqiangmler_sts24_task1
