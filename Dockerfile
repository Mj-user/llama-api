# Gunakan image python
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Salin file requirements.txt dan install dependency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file project ke dalam container
COPY . .

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
