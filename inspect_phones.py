import zipfile
import os

phone_dir = r'c:\Users\Win10\Desktop\Harness Bridge\Phones'
for fname in sorted(os.listdir(phone_dir)):
    if not fname.endswith('.zip'):
        continue
    z = zipfile.ZipFile(os.path.join(phone_dir, fname))
    with z.open('Metadata.csv') as f:
        meta = f.read().decode('utf-8').strip().split('\n')
        headers = [h.strip() for h in meta[0].split(',')]
        values = [v.strip() for v in meta[1].split(',')]
        d = dict(zip(headers, values))
    with z.open('Accelerometer.csv') as f:
        lines = f.read().decode('utf-8').strip().split('\n')
    device = d.get('device name', '?')
    rows = len(lines) - 1
    sr = d.get('sampleRateMs', '?')
    print(fname + ': device=' + device + ' rows=' + str(rows) + ' sampleRateMs=' + str(sr))
