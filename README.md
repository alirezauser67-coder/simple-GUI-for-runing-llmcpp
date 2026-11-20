# رابط گرافیکی llama.cpp

> [English version](README_en.md)

یک برنامه دسکتاپ ساده برای اجرای `llama-server` با مدل‌های محلی GGUF؛ بدون نیاز به رابط وب.

## امکانات

- اجرای `llama-server` با تنظیماتی مثل context، threads، batch، flash attention و KV cache
- تنظیمات پیشنهادی براساس اطلاعات واقعی فایل GGUF
- نمایش وضعیت GPU، RAM، CPU و دیسک، همراه با فهرست پردازه‌ها
- اجرای prompt و نمایش پاسخ، زمان پاسخ و سرعت تولید توکن
- نمودار سرعت توکن و پشتیبانی از رابط فارسی و انگلیسی

## پیش‌نیازها

- Windows 10 یا 11
- Python 3.9 یا جدیدتر، همراه Tkinter
- فایل `llama-server.exe` از [llama.cpp](https://github.com/ggml-org/llama.cpp/releases)
- یک فایل مدل با فرمت `.gguf`
- درایور NVIDIA برای نمایش اطلاعات کارت گرافیک

## اجرا

```powershell
python -m pip install -r requirements.txt
python llama_runner.py
```

نصب `psutil` اختیاری است، اما تشخیص CPU و RAM را دقیق‌تر می‌کند.

## راهنمای سریع

1. در زبانه **Server Setup** مسیر `llama-server.exe` و فایل مدل را انتخاب کنید.
2. یک preset انتخاب کنید یا از تحلیل خودکار مدل استفاده کنید.
3. سرور را روشن کنید.
4. در زبانه **Run Task** یک prompt بنویسید و اجرا کنید.
5. نتیجه و سرعت توکن را در **Task Inspector** ببینید.

| زبانه | کاربرد |
|---|---|
| Server Setup | انتخاب مدل، تنظیم سرور و مشاهده لاگ |
| GPU Processes | مانیتور سخت‌افزار و پردازه‌ها |
| Run Task | اجرای prompt و دریافت پاسخ |
| Task Inspector | نمودار سرعت توکن |

## مشکلات رایج

- **GPU نمایش داده نمی‌شود:** درایور NVIDIA را نصب یا به‌روزرسانی کنید و وجود `nvidia-smi` را بررسی کنید.
- **سرور اجرا نمی‌شود:** مسیر فایل exe، لاگ سرور و میزان حافظه کارت گرافیک را بررسی کنید. در صورت نیاز context یا batch را کمتر کنید.
- **مدل شناسایی نمی‌شود:** مطمئن شوید فایل واقعاً GGUF و سالم است.

## فایل‌ها

- `llama_runner.py`: کل برنامه
- `requirements.txt`: وابستگی اختیاری `psutil`
