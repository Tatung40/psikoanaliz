"""Strict input checks around the unchanged answer-key scoring API."""
import io
import pandas as pd
from core.data_processing import load_dataset


def validate_key(key,items,data=None):
    if len(items)!=len(set(items)):raise ValueError('Madde listesi benzersiz olmalıdır.')
    if set(key)!=set(items):
        raise ValueError(f'Anahtar madde adları tam eşleşmeli. Eksik: {sorted(set(items)-set(key))}; fazla: {sorted(set(key)-set(items))}')
    key={str(k):str(v).strip().upper() for k,v in key.items()}
    if any(v not in 'ABCDE' or len(v)!=1 for v in key.values()):raise ValueError('Cevap anahtarı yalnız A/B/C/D/E olabilir.')
    if data is not None:
        raw=data[items].astype('string').apply(lambda c:c.str.strip().str.upper())
        valid=raw.isin(list('ABCDE')) | raw.isna() | raw.eq('')
        if not valid.all().all():raise ValueError('Ham yanıtlarda A–E dışında değer var; sayısal veriyi ABCDE olarak puanlamayın.')
    return key


def read_key(source,file_type,items,data=None):
    if file_type=='xlsx':
        buffer=io.BytesIO(source) if isinstance(source,bytes) else source
        book=pd.ExcelFile(buffer)
        sheet='cevap_anahtari' if 'cevap_anahtari' in book.sheet_names else book.sheet_names[0]
        frame=pd.read_excel(book,sheet_name=sheet)
    else:frame,_=load_dataset(source,file_type=file_type)
    names={str(c).strip().casefold():c for c in frame}
    if len(frame.columns)!=2 or set(names)!={'madde','dogru_cevap'}:raise ValueError('Anahtar sütunları yalnız madde, dogru_cevap olmalıdır; yinelenen başlık kullanmayın.')
    keys=frame[names['madde']].astype('string').str.strip()
    if keys.isna().any() or keys.duplicated().any():raise ValueError('Anahtarda boş veya yinelenen madde adı var.')
    return validate_key(dict(zip(keys,frame[names['dogru_cevap']])),items,data)
