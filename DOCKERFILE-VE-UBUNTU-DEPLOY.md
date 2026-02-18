# Dockerfile İncelemesi ve Ubuntu Sunucuya Deploy Rehberi

## 1. Mevcut Dockerfile Detaylı İnceleme

### Genel yapı
- **Multi-stage build:** İki aşama (builder + runtime). Son imajda sadece çalışma zamanı bileşenleri kalır, boyut küçülür.
- **Builder:** Go uygulamasını derler.
- **Runtime:** Alpine tabanlı, sadece binarı ve kütüphaneleri içerir.

---

### Stage 1 – Builder (`golang:1.21-alpine`)

| Satır / Bölüm | Açıklama |
|---------------|----------|
| `FROM golang:1.21-alpine` | Go 1.21 ile Alpine Linux; derleme ortamı. |
| `apk add git build-base pkgconfig opencv-dev cmake` | CGO ile derleme ve OpenCV bağlantısı için gerekli paketler. Proje şu an CGO kullanmıyor (sadece `exec` ile Python çağrılıyor), bu paketler eski/alternatif bir mimari için. |
| `WORKDIR /app` | Çalışma dizini. |
| `COPY go.mod go.sum` | Bağımlılıkları indirmek için önce mod dosyaları kopyalanır (katman önbelleği). |
| `go mod download` | Go modülleri indirilir. |
| `COPY . .` | Tüm proje kopyalanır (kaynak + `scripts/`). |
| `CGO_ENABLED=1 ... go build ... -o /app/ocr-server ./cmd/server` | `ocr-server` binarı üretilir. CGO=1 OpenCV/ONNX için; mevcut kodda Go tarafında CGO kullanılmıyor. |

---

### Stage 2 – Runtime (`alpine:latest`)

| Satır / Bölüm | Açıklama |
|---------------|----------|
| `FROM alpine:latest` | Küçük son imaj. |
| `apk add opencv libstdc++ ca-certificates wget` | OpenCV runtime, standart C++ kütüphanesi, sertifikalar, wget (healthcheck için). |
| **ONNX Runtime kurulumu** | `wget` ile ONNX Runtime 1.16.3 indirilir, `/usr/local/lib` ve `include` dizinlerine kopyalanır, `ldconfig` çalıştırılır. |
| `WORKDIR /app` | Uygulama dizini. |
| `COPY --from=builder /app/ocr-server .` | **Sadece** Go binarı kopyalanır. `scripts/`, `models/`, vb. kopyalanmıyor. |
| `mkdir -p /app/models` | Model dizini oluşturulur. |
| `ENV PORT=8080 MODEL_PATH=... GIN_MODE=release LD_LIBRARY_PATH=...` | Varsayılan port 8080, model yolu, Gin release modu, ONNX kütüphane yolu. |
| `EXPOSE 8080` | Port 8080 dışarı açılır. |
| `HEALTHCHECK` | 30 saniyede bir `http://localhost:8080/health` kontrol edilir. |
| `CMD ["./ocr-server"]` | Sadece `./ocr-server` çalıştırılır. |

---

### Önemli uyumsuzluk (mevcut kodla)

Bu projede OCR işi **Go binarı değil**, **Python script** (`scripts/ocr_inference.py`) tarafından yapılıyor:

- Go sunucu başlarken `python3 scripts/ocr_inference.py --server --port 5555` çalıştırıyor.
- Python tarafında EasyOCR / TrOCR (PyTorch, transformers) kullanılıyor; ONNX kullanılmıyor.

Mevcut Dockerfile’da ise:

1. **Python yok** – Alpine imajında `python3` yüklü değil.
2. **`scripts/` kopyalanmıyor** – Runtime stage’de sadece `ocr-server` var; `ocr_inference.py` container içinde yok.
3. **ONNX kullanılmıyor** – Uygulama Python/EasyOCR/TrOCR kullandığı için ONNX kurulumu bu mimari için gereksiz.

Sonuç: Bu Dockerfile ile üretilen imajı çalıştırınca `python3` veya script bulunamadığı için OCR API düzgün çalışmaz. Ubuntu’ya deploy için ya bu Dockerfile’ı “Python + script” içerecek şekilde değiştirmeniz ya da Docker kullanmadan doğrudan Ubuntu’da çalıştırmanız gerekir.

---

## 2. Ubuntu Sunucuya Deploy – Yapılacaklar Listesi

### Seçenek A: Docker ile (önerilen: düzeltilmiş Dockerfile)

Aşağıdaki adımlar, **Python + Go** içeren ve mevcut koda uygun bir imaj ile deploy için.

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
   Proje kökünde **Python + Go** içeren `Dockerfile.python` kullanın (mevcut `Dockerfile` ONNX’e göre, uygulama Python kullanıyor):
   ```bash
   cd /path/to/OCR-main
   docker build -f Dockerfile.python -t ocr-api:latest .
   ```
   İlk build (PyTorch indirmesi) 10–20 dakika sürebilir.

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
- [ ] **Docker ile:** Düzeltilmiş Dockerfile ile imaj build edildi; `scripts/` ve Python runtime imajda
- [ ] **Native:** `go build`, venv aktif, gerekirse `download_models.sh` çalıştırıldı
- [ ] Port (8082 veya seçtiğiniz) açıldı ve uygulama bu portta dinliyor
- [ ] `curl http://<sunucu>:8082/health` ile sağlık kontrolü yapıldı
- [ ] Firewall’da ilgili port açıldı
- [ ] İsteğe bağlı: systemd servisi veya `restart: unless-stopped` ile sürekli çalışma ayarlandı

---

## 4. Mevcut Dockerfile ile Yapılmaması Gerekenler

- Mevcut `Dockerfile` (ONNX + sadece Go binarı) ile production’da **doğrudan** deploy etmeyin; OCR çalışmaz.
- Portu 8082 yapmak için runtime’da `-e PORT=8082` ve `-p 8082:8082` kullanın; Dockerfile’daki `EXPOSE`/`ENV PORT` ile uyumlu olsun.

Bu rehber, Dockerfile’ı detaylı incelemeniz ve Ubuntu’ya adım adım deploy etmeniz için yeterli olmalı. İsterseniz bir sonraki adımda projeye `Dockerfile.python` (Go + Python + scripts içeren çalışan örnek) ekleyebilirim.
