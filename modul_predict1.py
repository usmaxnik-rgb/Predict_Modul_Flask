from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import StandardScaler
import pickle
import os
from sklearn.svm import SVR

print('Импорт выполнен успешно')


# Загрузка моделей
try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    print('Scaler загружен успешно')
    print(f'Тип scaler: {type(scaler)}')  # Эта строка покажет проблему


    svr = pickle.load(open('modul_predict_model_svr_english.pkl', 'rb'))
    print('Модель SVR загружена успешно')
except Exception as e:
    print(f'Ошибка загрузки моделей: {e}')
    raise

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('main1.html')
    
    if request.method == 'POST':
        try:
            # Собираем и преобразуем данные
            X_from_form = [
                float(request.form['angle']),
                float(request.form['step']),
                float(request.form['density']),
                float(request.form['matrix_ratio']),
                float(request.form['material_density']),
                float(request.form['elasticity']),
                float(request.form['hardener']),
                float(request.form['epoxy_content']),
                float(request.form['flash_point']),
                float(request.form['surface_density']),
                float(request.form['tensile_strength']),
                float(request.form['resin_consumption'])
            ]
            
            print('Входной вектор X:', X_from_form)
            
            # Масштабирование
            X_scaled = scaler.transform([X_from_form])
            
            # Прогноз
            prediction = svr.predict(X_scaled)
            result = float(prediction[0])
            
            print('Результат прогноза:', result)
            
            return render_template('main1.html', result=result)
            
        except KeyError as e:
            return f"Ошибка: отсутствует поле формы: {e}"
        except ValueError as e:
            return f"Ошибка ввода данных: убедитесь, что все поля содержат числа. {e}"
        except Exception as e:
            return f"Неизвестная ошибка: {e}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)