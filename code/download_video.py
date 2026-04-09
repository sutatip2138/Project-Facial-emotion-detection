import yt_dlp

url = 'https://www.youtube.com/watch?v=9C7gMOi_CBc'
ydl_opts = {
    'format': 'mp4',
    'outtmpl': 'C:/Users/Admin/Desktop/pro ject/videos/%(title)s.%(ext)s',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("ดาวน์โหลดสำเร็จ!")


