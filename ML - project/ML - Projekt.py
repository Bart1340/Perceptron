#!/usr/bin/env python
# coding: utf-8

# # Sprawdzenie i oczyszczenie danych

# In[1]:


#Importuję potrzebne biblioteki. 
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#Importuję zbiór danych.
df1 = pd.read_excel("Dataset Basic.xlsx")


# In[2]:


#Sprawdzam jak wygląda zbiór danych.
#df1.head(1000)


# In[3]:


#Sprawdzam podstawowe statystyki dotyczące zmiennych.
df1.describe()


# In[4]:


df1.info()


# In[5]:


df1 = df1.drop('Z13',axis=1) #Usuwam zmienną Z13, ponieważ nie będzie wykorzystywana w sieci. 
df1 = df1.drop([0,1]) #Usuwam pierwsze dwa wiersze, które służą jako opis dla zmiennych.

df1['Z9'] = df1['Z9'].astype(bool) #Zmieniam typ zmiennych, które przyjmują wartości 0 lub 1 na boolean (wartości logiczne - prawda lub fałsz).
df1['Z10'] = df1['Z9'].astype(bool)
df1['Z12'] = df1['Z9'].astype(bool)

#Po przeglądnięciu zbioru postanowiłem, że najlepszym sposobem na pozbycie się brakujących danych będzie zamienienie ich na 0.
#Przyjmuję więc, że brak danych jest równoznaczny z tym, że reakcji/obrazu po prostu nie było.
df1['Z11'].fillna(0, inplace=True) 
df1['Z14'].fillna(0, inplace=True)
df1['Z15'].fillna(0, inplace=True)
df1['Z16'].fillna(0, inplace=True)
df1['Z17'].fillna(0, inplace=True)
df1['Z18'].fillna(0, inplace=True)
df1['Z19'].fillna(0, inplace=True)
df1['Z20'].fillna(0, inplace=True)
df1['Z21'].fillna(0, inplace=True)

#Resetuję index w związku z usunięciem dwóch pierwszych wierszy.
df1 = df1.reset_index() 
df1 = df1.drop('index',axis=1)


# In[6]:


#Tak wygląda gotowy zbiór danych.
#df1.head(1000)


# In[7]:


df1.describe()


# In[8]:


df1.info()


# In[9]:


#Dodatkowo sprawdzam w Excelu czy wszystko się zgadza.
df1.to_excel('test.xlsx')


# # Pierwsza Iteracja

# ### Wczytywanie danych do modelu

# In[10]:


#Konwertuję zmienne na tablicę numpy.
X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values #Zmienne na podstawie których przewidujemy (cechy wejściowe). 
y = df1['Z21'].values #Liczba komentarzy, którą chcemy przewidywać (wektor docelowy).


# ### Podział danych na zbiór treningowy i testowy

# In[11]:


#Zbiór treningowy - 80%, Zbiór testowy - 20%, 'random_state=42' zapewnia powtarzalność podziału danych.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Skalowanie zmiennych

# In[12]:


#Standaryzuję zbiór treningowy za pomocą StandardScaler z biblioteki sklearn. 
#Następnie przeprowadzam takie samo skalowanie na zbiorze testowym, zapewniając spójność między obydwoma zbiorami.
#Standardyzacja przekształca dane tak, aby miały średnią równą zero i wariancję równą jeden.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ### Przygotowanie architektury sieci (Perceptron)
# Definiuje klasę 'Perceptron', która implementuje algorytm perceptronu złożony z pojedynczego sztucznego neuronu. Jego celem jest przewidywanie liczby komentarzy na podstawie podanych zmiennych. 

# In[13]:


class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100): 
        self.num_features = num_features #Liczba cech wejściowych.
        self.learning_rate = learning_rate #Współczynnik uczenia (jak szybko model będzie się dostosowywał do danych treningowych).
        self.num_epochs = num_epochs #Licza epok (ile razy model przejdzie przez cały zbiór danych treningowych podczas procesu uczenia).
        self.weights = np.zeros(num_features + 1) #Wagi początkowe - wektor zer o długości = liczba cech + 1 (dodatkowa jedynka odnosi się do wagi biasu)
        print("Wagi początkowe:")
        print(self.weights) #Wypisuje wagi początkowe modelu.
        
    def get_weights(self):
        return self.weights #Zwraca aktualne wagi modelu.
    
    def activate(self, x):
            return x #Funkcja aktywacji, która zwraca input (x) bez żadnych zmian.

    def predict(self, x): #Służy do przewidywania wyniku na podstawie podanego wektora wejściowego x.
        x = np.insert(x, 0, 1)  #Dodanie 1 na pozycji zerowej odpowiada za bias.
        activation = np.dot(self.weights, x) #Funkcja np.dot z biblioteki NumPy wykonuje iloczyn skalarany między wektorem wag, a wektorem x.
        return activation #Wynik zwraca wartość liczbową (aktywację), która stanowi naszą predykcję.
    
    def train(self, X, y): #Trenowanie perceptronu poprzez aktualizację wag na podstawie błędu predykcji.
        for _ in range(self.num_epochs): #Pętla iterująca przez określoną liczbę epok.
            for i in range(len(X)): #Pętla iterująca przez wszystkie przykłady treningowe ('i' to indeks bieżącego przykładu).
                x = X[i] #Wektor wejściowy (x) dla danego przykładu.
                y_true = y[i] #Wartość docelowa dla danego przykładu.
                y_pred = self.predict(x) #Obliczenie predykcji dla danego x. 
                error = y_true - y_pred #Obliczenie błędu predykcji jako różnicy między wartością docelową 'y_true' a predykcją 'y_pred'.
                self.weights += self.learning_rate * error * np.insert(x, 0, 1) #Aktualizacja wag perceptronu na podstawie błędu predykcji.


# ### Inicjalizacja i trening perceptronu

# In[14]:


perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100) #Ustawia liczbę cech na równą liczbie kolumn w zbiorze treningowym + określa współczynnik uczenia i liczbę epok. 
perceptron.train(X_train_scaled, y_train) #Uczenie modelu za pomocą metody 'train' poprzez przekazanie standaryzowanych danych treningowych oraz danych docelowych.


# ### Prognozowanie na danych treningowych i testowych

# In[15]:


#Generuje przewidywane wartości na podstawie wytrenowanego perceptronu dla danych treningowych i testowych.
y_pred_train = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test = [perceptron.predict(x) for x in X_test_scaled]
#print(y_pred_train) #lista przewidywanych wartości dla danych treningowych.
#print(y_pred_test) #lista przewidywanych wartości dla danych testowych.


# ### Obliczanie błędu średniokwadratowego (MSE)

# In[16]:


#Miara oceny jakości modelu
#Oblicza różnicę pomiędzy przewidywanymi wartościami a rzeczywistymi wartościami osobno w zbiorze treningowym i zbiorze testowym.
#Różnice zostają podniesione do kwadratu i obliczana jest ich średnia wartość.
mse_train = np.mean((y_pred_train - y_train) ** 2) 
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (treningowe):', mse_train)
print('Błąd (testowe):', mse_test)


# Wartości błędu są bardzo wysokie, co sprawia, że obecna architektura nie będzie przydatna w tworzeniu predykcji. 

# ### Wagi

# In[17]:


trained_weights = perceptron.get_weights() #pobiera wagi modelu, które są rezultatem procesu uczenia (wraz z wagą biasu).
print("Wagi końcowe:")
print(trained_weights)


# Wartości wag reprezenują wpływ (negatywny lub pozytywny) danej zmiennej na liczbę komentarzy oraz siłę tego wpływu.

# ### Przykład działania na nowych danych

# In[18]:


#Nowe dane
new_data = np.array([[False, True, 3, False, 11, 50, 5, 23, 1, 2, 1]])

#Standaryzacja nowych danych
new_data_scaled = scaler.transform(new_data)

#Przewidywanie liczby komentarzy
prediction = perceptron.predict(new_data_scaled)

#Wyświetlenie przewidywanej liczby komentarzy
print('Przewidywana liczba komentarzy:', prediction)


# Przewidywana liczba komentarzy jest w oczywisty sposób nietrafiona.

# # Kolejne Iteracje - wykaz sprawdzonych architektur sieci

# ## Badanie wpływu zmiany typu inputów na wyniki

# ### Zmiana na Integer (liczby całkowite)

# In[19]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights
    
    def activate(self, x):
        return x
    
    #Dodanie funkcji, która konwertuje każdy input x na liczby całkowite (integer).
    def preprocess_input(self, x):
        processed_input = [int(feature) for feature in x]
        return processed_input
    
    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation  

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train)

y_pred_train = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test = [perceptron.predict(x) for x in X_test_scaled]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Ani wartości błędu ani wagi końcowe nie zmieniły się - wyniki nadal nie są przydatne.

# In[20]:


new_data = np.array([[False, True, 3, False, 11, 50, 5, 23, 1, 2, 1]])
new_data_scaled = scaler.transform(new_data)
prediction = perceptron.predict(new_data_scaled)
print('Przewidywana liczba komentarzy:', prediction)


# ### Zmiana na Float (liczby zmiennoprzecinkowe)

# In[21]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x
    
    #Dodanie funkcji, która konwertuje każdy input x na liczby zmiennoprzecinkowe (float).
    def preprocess_input(self, x):
        processed_input = [float(feature) for feature in x]
        return processed_input

    def predict(self, x): 
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train)

y_pred_train = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test = [perceptron.predict(x) for x in X_test_scaled]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Ani wartości błędu ani wagi końcowe nie zmieniły się - wyniki nadal nie są przydatne.

# In[22]:


new_data = np.array([[False, True, 3, False, 11, 50, 5, 23, 1, 2, 1]])
new_data_scaled = scaler.transform(new_data)
prediction = perceptron.predict(new_data_scaled)
print('Przewidywana liczba komentarzy:', prediction)


# ### Zmiana na String (wartość tekstowa)

# In[23]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x
    
    #Dodanie funkcji, która konwertuje każdy input x na wartość tekstową (string).
    def preprocess_input(self, x):
        processed_input = [string(feature) for feature in x]
        return processed_input

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train)

y_pred_train = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test = [perceptron.predict(x) for x in X_test_scaled]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Ani wartości błędu ani wagi końcowe nie zmieniły się - wyniki nadal nie są przydatne.

# In[24]:


new_data = np.array([[False, True, 3, False, 11, 50, 5, 23, 1, 2, 1]])
new_data_scaled = scaler.transform(new_data)
prediction = perceptron.predict(new_data_scaled)
print('Przewidywana liczba komentarzy:', prediction)


# ## Badanie wpływu zmiany ilości inputów na wyniki

# ### Inputy związane z obrazkami i grafiką 

# In[25]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12']].values #Wybiera tylko pierwsze cztery zmienne jako input.
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train)

y_pred_train = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test = [perceptron.predict(x) for x in X_test_scaled]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Wartość błędu jest znacznie niższa niż w poprzednich przypadkach, co sprawia, że można użyć zmiennych dotyczących grafiki i obrazków w przewidywaniu liczby komentarzy.
# 
# Wagi końcowe wskazują, że zmienna logiczna dotycząca obecności (lub braku) grafiki ma decydujący (pozytywny) wpływ na liczbę komentarzy. Nieznacznie negatywny wpływ ma natomiast fakt, że obrazki są tylko online. Reszta zmiennych nie ma większego znaczenia. 

# In[26]:


new_data = np.array([[False, True, 3, False]])
new_data_scaled = scaler.transform(new_data)
prediction = perceptron.predict(new_data_scaled)
print('Przewidywana liczba komentarzy:', prediction)


# Przewidywana liczba komentarzy wydaje się być znacznie bardziej prawdopodobna.

# ### Inputy związane z reakcjami

# In[27]:


X = df1[['Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values #Wybiera tylko zmienne związane z reakcjami jako input.
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x
    
    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train)

y_pred_train = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test = [perceptron.predict(x) for x in X_test_scaled]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Wartości błędu są bardzo wysokie, co sprawia, że ta architektura nie będzie przydatna w tworzeniu predykcji.

# In[28]:


new_data = np.array([[32, 52, 12, 14, 1, 2, 1]])
new_data_scaled = scaler.transform(new_data)
prediction = perceptron.predict(new_data_scaled)
print('Przewidywana liczba komentarzy:', prediction)


# ## Badanie wpływu zmiany sposobu generowania wag początkowych na wyniki

# ### Losowe wartości z rozkładu jednorodnego

# In[29]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        #Nowe wagi początkowe.
        #Generuje losowe liczby z rozkładu jednorodnego między -1 a 1, zwracając tablicę o rozmiarze = liczba cech + 1.
        self.weights = np.random.uniform(-1, 1, size=num_features + 1) 
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train)

y_pred_train = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test = [perceptron.predict(x) for x in X_test_scaled]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Wartości błędu są bardzo wysokie, co sprawia, że ta architektura nie będzie przydatna w tworzeniu predykcji.

# In[30]:


new_data = np.array([[False, True, 3, False, 11, 50, 5, 23, 1, 2, 1]])
new_data_scaled = scaler.transform(new_data)
prediction = perceptron.predict(new_data_scaled)
print('Przewidywana liczba komentarzy:', prediction)


# ### Losowe wartości z rozkładu normalnego

# In[31]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        #Nowe wagi początkowe.
        #Generuje losowe liczby z rozkładu normalnego o średniej równiej 0 i odchyleniu standardowym równym 1, zwracając tablicę o rozmiarze = liczba cech + 1.
        self.weights = np.random.normal(0, 1, size=num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train)

y_pred_train = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test = [perceptron.predict(x) for x in X_test_scaled]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Wartości błędu są bardzo wysokie, co sprawia, że ta architektura nie będzie przydatna w tworzeniu predykcji.

# In[32]:


new_data = np.array([[False, True, 3, False, 11, 50, 5, 23, 1, 2, 1]])
new_data_scaled = scaler.transform(new_data)
prediction = perceptron.predict(new_data_scaled)
print('Przewidywana liczba komentarzy:', prediction)


# ### Metoda Xavier/Glorot 

# In[33]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        #Nowe wagi początkowe.
        limit = np.sqrt(6 / (num_features + 1)) #Limit wartości wag na podstawie liczby cech wejściowych.
        self.weights = np.random.uniform(-limit, limit, size=num_features + 1) #Generuje losowe wagi z rozkładu jednostajnego w przedziale [-limit, limit].
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train)

y_pred_train = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test = [perceptron.predict(x) for x in X_test_scaled]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Wartości błędu są bardzo wysokie, co sprawia, że ta architektura nie będzie przydatna w tworzeniu predykcji.

# In[34]:


new_data = np.array([[False, True, 3, False, 11, 50, 5, 23, 1, 2, 1]])
new_data_scaled = scaler.transform(new_data)
prediction = perceptron.predict(new_data_scaled)
print('Przewidywana liczba komentarzy:', prediction)


# ## Badanie wpływu zmiany sposobu przetwarzania danych wejściowych na wyniki

# ### Normalizacja danych wejściowych
# Zastosowanie funkcji MinMaxScaler jako skalera zamiast funkcji StandardScaler.

# In[35]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Wykonuje normalizację danych przy użyciu skalera typu MinMaxScaler z biblioteki scikit-learn.
#Przekształca wartości danych w taki sposób, aby mieściły się w zakresie (0, 1).
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

perceptron = Perceptron(num_features=X_train_normalized.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_normalized, y_train)

y_pred_train = [perceptron.predict(x) for x in X_train_normalized]
y_pred_test = [perceptron.predict(x) for x in X_test_normalized]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Wartość błędu jest najniższa ze wszystkich dotychczasowych przypadków.
# 
# Wagi końcowe pokazują następujące zależności:
# - Największy pozytywny wpływ na liczbę komentarzy mają zmienne dotyczące reakcji "LOVE", "HAHA" i "WRR".
# - Mniejszy pozytywny wpływ mają zmienne dotyczące obecności grafiki oraz reakcji "CRY", "WOW" i "HUG".
# - Największy negatywny wpływ na liczbę komentarzy ma zmienna, która wskazuję na to czy obrazki są dostępne tylko online.
# - Wpływ reszty zmiennych jest stosunkowo marginalny.

# In[36]:


new_data = np.array([[False, True, 3, False, 11, 50, 5, 23, 1, 2, 1]])
new_data_scaled = scaler.transform(new_data)
prediction = perceptron.predict(new_data_scaled)
print('Przewidywana liczba komentarzy:', prediction)


# Na tle dotychczasowych iteracji, wynik proponowany przez tę architekturę wydaję się być najbardziej realistyczny. 

# ### Przekształcenie logarytmiczne danych wejściowych

# In[37]:


df1['Z9'] = df1['Z9'].astype('int64')
df1['Z10'] = df1['Z9'].astype('int64')
df1['Z12'] = df1['Z9'].astype('int64')

X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Logarytmowanie danych treningowych i testowych za pomocą funkcji log1p z biblioteki NumPy.
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

perceptron = Perceptron(num_features=X_train_log.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_log, y_train)

y_pred_train = [perceptron.predict(x) for x in X_train_log]
y_pred_test = [perceptron.predict(x) for x in X_test_log]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Przekształcenie logarytmiczne również wydaje się znacznie poprawiać otrzymywane wyniki. W porównaniu do normalizacji ten sposób przetwarzania danych wejściowych nieznacznie zmniejsza błąd w danych treningowych, jednocześnie nieznacznie zwiększając błąd w danych testowych.
# 
# Wagi wskazują, że na liczbę komentarzy pozytywne wpływa przede wszystkim ilość reakcji "WRR", a w dalszej kolejność ilość reakcji "HAHA", "LOVE" oraz obecność grafiki. 

# In[38]:


new_data = np.array([[False, True, 3, False, 11, 50, 5, 23, 1, 2, 1]])
new_data_log = np.log1p(new_data)
prediction = perceptron.predict(new_data_log)
print('Przewidywana liczba komentarzy:', prediction)


# Przewidywana liczba komentarzy jest dwa razy większa niż w poprzedniej architekturze.

# ## Badanie wpływu zmiany sposobu przetwarzania danych wyjściowych na wyniki

# ### Standaryzacja

# In[39]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

#Standaryzacja danych wyjściowych
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train_scaled)

y_pred_train = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test = [perceptron.predict(x) for x in X_test_scaled]

mse_train = np.mean((y_pred_train- y_train) ** 2)
mse_test = np.mean((y_pred_test- y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Wartości błędu są bardzo wysokie, co sprawia, że ta architektura nie będzie przydatna w tworzeniu predykcji.

# ### Normalizacja 

# In[40]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

#Normalizacja danych wyjściowych
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train_scaled)

y_pred_train_scaled = np.array([perceptron.predict(x) for x in X_train_scaled])
y_pred_test_scaled = np.array([perceptron.predict(x) for x in X_test_scaled])

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Wartości błędu są bardzo wysokie, co sprawia, że ta architektura nie będzie przydatna w tworzeniu predykcji.

# ### Przekształcenie logarytmiczne

# In[41]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

#Logarytmowanie danych wyjściowych
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train_log)

y_pred_train_log = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test_log = [perceptron.predict(x) for x in X_test_scaled]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Wartości błędu są bardzo wysokie, co sprawia, że ta architektura nie będzie przydatna w tworzeniu predykcji.

# ## Badanie wpływu zmiany sposobu przetwarzania danych wejściowych i wyjściowych na wyniki

# ### Normalizacja danych wejściowych i wyjściowych

# In[42]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalizacja danych wejściowych
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

#Normalizacja danych wyjściowych
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train_scaled)

y_pred_train_scaled = np.array([perceptron.predict(x) for x in X_train_scaled])
y_pred_test_scaled = np.array([perceptron.predict(x) for x in X_test_scaled])

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Wartości błędu są bardzo wysokie, co sprawia, że ta architektura nie będzie przydatna w tworzeniu predykcji.

# ### Przekształcenie logarytmiczne wejściowych i wyjściowych

# In[43]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Logarytmowanie danych wejściowych
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

#Logarytmowanie danych wyjściowych
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

perceptron = Perceptron(num_features=X_train_scaled.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_scaled, y_train_log)

y_pred_train_log = [perceptron.predict(x) for x in X_train_scaled]
y_pred_test_log = [perceptron.predict(x) for x in X_test_scaled]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Wartości błędu są bardzo wysokie, co sprawia, że ta architektura nie będzie przydatna w tworzeniu predykcji.

# # Finalna architektura

# In[44]:


X = df1[['Z9', 'Z10', 'Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']].values
y = df1['Z21'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalizacja danych wejściowych przy użyciu skalera typu MinMaxScaler.
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        #Losowe wartości z rozkładu normalnego jako wagi początkowe.
        self.weights = np.random.normal(0, 1, size=num_features + 1)
        print("Wagi początkowe:")
        print(self.weights)
        
    def get_weights(self):
        return self.weights

    def activate(self, x):
        return x

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        activation = np.dot(self.weights, x)
        return activation

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)

perceptron = Perceptron(num_features=X_train_normalized.shape[1], learning_rate=0.01, num_epochs=100)
perceptron.train(X_train_normalized, y_train)

y_pred_train = [perceptron.predict(x) for x in X_train_normalized]
y_pred_test = [perceptron.predict(x) for x in X_test_normalized]

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print('Błąd (Treningowe):', mse_train)
print('Błąd (Testowe):', mse_test)

trained_weights = perceptron.get_weights()
print("Wagi końcowe:")
print(trained_weights)


# Tak opracowana architektura ma najmniejszy błąd średniokwadratowy. <br>
# - Wykorzystuje normalizacje danych wejściowych za pomocą MinMaxScaler
# - Wagi początkowe to losowe wartości z rozkładu normalnego 

# In[45]:


new_data = np.array([[False, True, 3, False, 11, 50, 5, 23, 1, 2, 1]])
new_data_scaled = scaler.transform(new_data)
prediction = perceptron.predict(new_data_scaled)
print('Przewidywana liczba komentarzy:', prediction)


# ## Zestaw finalnie opracowanych wag
# 
# Wagi końcowe:
# [ 22.64486099  -5.34182676  -3.608122   -20.47909052  -5.73824676 <br>
#  179.46125369  20.25523492 160.01880244  14.75047674  36.46792395 <br>
#  156.41042624  -0.44806161]
#  
#  Wagi końcowe pokazują następujące zależności:
# - Największy pozytywny wpływ na liczbę komentarzy mają zmienne dotyczące reakcji "LOVE", "HAHA" i "WRR".
# - Mniejszy pozytywny wpływ mają zmienne dotyczące obecności grafiki oraz reakcji "CRY", "WOW" i "HUG".
# - Największy negatywny wpływ na liczbę komentarzy ma zmienna, która wskazuję na to czy obrazki są dostępne tylko online.
# - Wpływ reszty zmiennych jest stosunkowo marginalny.
