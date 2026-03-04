# Dockerfile İncelemesi ve Ubuntu Sunucuya Deploy Rehberi

## 1. Mevcut Dockerfile Detaylı İnceleme

### Genel yapı
- **Multi-stage build:** İki aşama (builder + runtime). Son imajda sadece çalışma zamanı bileşenleri kalır, boyut küçülür.
- **Builder:** Go uygulamasını derler.
- **Runtime:** Alpine tabanlı, sadece binarı ve kütüphaneleri içerir.

---

### Stage 1 – Builder (`golang:1.21-bookworm`)

| Satır / Bölüm | Açıklama |
|---------------|----------|
| `FROM golang:1.21-bookworm` | Go 1.21 ile Debian Bookworm tabanlı resmi imaj; derleme ortamı. |
| `apt-get install -y git build-essential pkg-config` | Go derlemesi için temel build araçları. CGO kullanılmadığı için ekstra OpenCV/ONNX paketleri yok. |
| `WORKDIR /app` | Çalışma dizini. |
| `COPY go.mod go.sum ./` | Bağımlılıkları indirmek için önce mod dosyaları kopyalanır (katman önbelleği). |
| `go mod download` | Go modülleri indirilir. |
| `COPY . .` | Tüm proje kopyalanır (kaynak + `scripts/`). |
| `CGO_ENABLED=0 GOOS=linux go build -o /app/ocr-server ./cmd/server` | Statik Go binarı üretilir; CGO devre dışı, OpenCV/ONNX bağımlılığı yok. |

---

### Stage 2 – Runtime (`python:3.10-slim-bookworm`)

| Satır / Bölüm | Açıklama |
|---------------|----------|
| `FROM python:3.10-slim-bookworm` | Hafif Python 3.10 tabanlı imaj; Tesseract/EasyOCR/TrOCR için Python runtime. |
| `apt-get install -y ... tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng` | Tesseract motoru ve Türkçe/İngilizce dil paketleri ile gerekli sistem kütüphaneleri kurulur. |
| `pip install pytesseract Pillow numpy opencv-python-headless ...` | Python tarafındaki OCR ve görüntü işleme bağımlılıkları (Tesseract + EasyOCR + TrOCR) kurulur. |
| `WORKDIR /app` | Uygulama dizini. |
| `COPY --from=builder /app/ocr-server .` | Go ile derlenmiş `ocr-server` binarı kopyalanır. |
| `COPY scripts/ ./scripts/` | Python OCR sunucusu (`scripts/ocr_inference.py`) imaja dahil edilir. |
| `mkdir -p /app/models /app/uploads` | Model ve upload dizinleri oluşturulur. |
| `ENV PORT=8080 MODEL_PATH=/app/models GIN_MODE=release ...` | Varsayılan port, model yolu ve performans ayarları environment değişkenleri ile set edilir. |
| `EXPOSE 8080` | Port 8080 dışarı açılır. |
| `HEALTHCHECK` | 30 saniyede bir `http://localhost:8080/health` kontrol edilir. |
| `CMD ["./ocr-server"]` | Go API sunucusu başlatılır; uygulama Python OCR sunucusuna HTTP üzerinden bağlanır. |

---

### Kod ile uyumluluk

Bu projede OCR işi **Go binarı değil**, **Python script** (`scripts/ocr_inference.py`) tarafından yapılıyor:

- Go sunucu başlarken `python3 scripts/ocr_inference.py --server --port 5555` çalıştırıyor.
- Python tarafında Tesseract + EasyOCR / TrOCR (PyTorch, transformers) kullanılıyor; ONNX artık kullanılmıyor.

**Güncel Dockerfile** tam olarak bu mimariye göre güncellendi:

1. **Python var** – Runtime imajı `python:3.10-slim-bookworm` tabanlı ve `python3` ile tüm gerekli Python paketlerini içeriyor.
2. **`scripts/` kopyalanıyor** – Runtime stage’de `scripts/ocr_inference.py` imaja kopyalanıyor ve Go tarafı bu scripti başlatıyor.
3. **ONNX yok** – Uygulama Python/Tesseract/EasyOCR/TrOCR kullandığı için ONNX Runtime kurulumu kaldırıldı; imaj daha sade ve küçük.

Sonuç: Mevcut Dockerfile ile üretilen imaj, OCR API’nin Go + Python mimarisiyle **uyumlu** ve production’da doğrudan kullanılabilir.

---

## 2. Ubuntu Sunucuya Deploy – Yapılacaklar Listesi

### Seçenek A: Docker ile (önerilen: mevcut Dockerfile)

Aşağıdaki adımlar, **Go + Python** içeren ve mevcut koda uygun Docker imajı ile deploy için.

#### 2.1 Sunucuda hazırlık

1. **Ubuntu güncellemesi**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Docker kurulumu**
   ```bash
   sudo apt install -y ca-certificates curl gnupg
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg
   echo "deb [arch=$(dpkg --print-architecture)] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt update
   sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   sudo usermod -aG docker $USER
   ```
   Çıkıp tekrar giriş yapın veya `newgrp docker` kullanın.

3. **Projeyi sunucuya almak**
   - Git ile: `git clone <repo-url> && cd OCR-main`
   - Veya proje klasörünü scp/rsync ile kopyalayın.

4. **Çalışan imajı build etmek**  
   Proje kökünde bulunan **güncel Dockerfile** (Go + Python) ile imajı build edin:
   ```bash
   cd /path/to/OCR-main
   docker build -t ocr-api:latest .
   ```
   İlk build (Python/Tesseract/EasyOCR/TrOCR kurulumu) birkaç dakika sürebilir.

5. **Model dizini (isteğe bağlı)**  
   İlk çalıştırmada Python modelleri indirilebilir; kalıcı olması için volume:
   ```bash
   mkdir -p ./models
   ```

6. **Container’ı çalıştırmak**
   ```bash
   docker run -d --name ocr-api -p 8082:8082 -v $(pwd)/models:/app/models -e PORT=8082 -e GIN_MODE=release --restart unless-stopped ocr-api:latest
   ```
   Portu (8082) ihtiyaca göre değiştirebilirsiniz.

7. **Kontrol**
   ```bash
   curl http://localhost:8082/health
   ```

8. **Firewall (varsa)**
   ```bash
   sudo ufw allow 8082/tcp
   sudo ufw reload
   ```

---

### Seçenek B: Docker kullanmadan (native Ubuntu)

Docker kullanmak istemezseniz, uygulamayı doğrudan Ubuntu’da çalıştırabilirsiniz.

#### 2.1 Go kurulumu

```bash
sudo apt update
sudo apt install -y golang-go
go version
```

#### 2.2 Python 3 ve bağımlılıklar

```bash
sudo apt install -y python3 python3-pip python3-venv
cd /path/to/OCR-main
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch transformers Pillow easyocr opencv-python-headless scipy scikit-image
```

#### 2.3 Model indirme (isteğe bağlı)

```bash
# Proje kökünde
./scripts/download_models.sh   # Varsa
# veya ilk çalıştırmada otomatik indirilir
```

#### 2.4 Proje modüllerini indirme ve derleme

```bash
cd /path/to/OCR-main
go mod download
go build -o ocr-server ./cmd/server
```

#### 2.5 Çalıştırma

```bash
export PORT=8082
export GIN_MODE=release
./ocr-server
```

Arka planda çalıştırmak için:

```bash
nohup ./ocr-server >> server.log 2>&1 &
# veya systemd servisi (aşağıda örnek)
```

#### 2.6 Systemd servisi (opsiyonel)

`/etc/systemd/system/ocr-api.service`:

```ini
[Unit]
Description=OCR API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/path/to/OCR-main
Environment="PORT=8082"
Environment="GIN_MODE=release"
Environment="PATH=/path/to/OCR-main/venv/bin:/usr/local/bin:/usr/bin"
ExecStart=/path/to/OCR-main/venv/bin/python3 -c "import subprocess; import os; os.chdir('/path/to/OCR-main'); subprocess.run(['./ocr-server'])"
Restart=unless-stopped

[Install]
WantedBy=multi-user.target
```

Daha basit alternatif: `ExecStart=/path/to/OCR-main/ocr-server` ve `WorkingDirectory` ile birlikte, Python’un sistemde veya venv’de `python3` olarak bulunması (PATH’te venv önde olacak şekilde).

```ini
ExecStart=/path/to/OCR-main/ocr-server
```

Sonra:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ocr-api
sudo systemctl start ocr-api
sudo systemctl status ocr-api
```

#### 2.7 Firewall

```bash
sudo ufw allow 8082/tcp
sudo ufw reload
```

---

## 3. Özet Kontrol Listesi (Ubuntu Deploy)

- [ ] Ubuntu güncellendi (`apt update && upgrade`)
- [ ] **Docker ile:** Docker ve (isteğe bağlı) Docker Compose kuruldu
- [ ] **Native:** Go + Python3 + venv + pip bağımlılıkları (torch, transformers, easyocr, opencv, vb.) kuruldu
- [ ] Proje sunucuya kopyalandı
- [ ] **Docker ile:** Mevcut Dockerfile (Go + Python) ile imaj build edildi; `scripts/` ve Python runtime imajda
- [ ] **Native:** `go build`, venv aktif, gerekirse `download_models.sh` çalıştırıldı
- [ ] Port (8082 veya seçtiğiniz) açıldı ve uygulama bu portta dinliyor
- [ ] `curl http://<sunucu>:8082/health` ile sağlık kontrolü yapıldı
- [ ] Firewall’da ilgili port açıldı
- [ ] İsteğe bağlı: systemd servisi veya `restart: unless-stopped` ile sürekli çalışma ayarlandı

---

## 4. Dockerfile ile İlgili Notlar

- Eski ONNX + sadece Go binarı içeren Dockerfile artık kullanılmıyor; repodaki **güncel** Dockerfile Go + Python mimarisiyle uyumludur ve production için kullanılabilir.
- Portu 8082 yapmak için runtime’da `-e PORT=8082` ve `-p 8082:8082` kullanın; Dockerfile’daki `EXPOSE`/`ENV PORT` ile uyumlu olsun.

Bu rehber, Dockerfile’ı detaylı incelemeniz ve Ubuntu’ya adım adım deploy etmeniz için yeterli olmalı.
