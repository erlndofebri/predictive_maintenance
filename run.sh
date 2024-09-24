python src/data_pipeline.py
python src/data_preprocessing.py
python src/modeling.py
sleep 5
python src/api2.py &
sleep 5
streamlit run src/main.py
