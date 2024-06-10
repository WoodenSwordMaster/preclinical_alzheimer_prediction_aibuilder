# preclinical_alzheimer_prediction_aibuilder
ส่งงาน ai builder สิ่งที่เอไอตัวนี้จะทำคือการทำภาพถ่าย mri scan ของสมองแบบ 3d ของผู้ป่วยเข้าไปแล้วจะบอกว่าผู้ป่วย เป็นโรคอัลไซเมอร์ ภาวะหลงลืมๆทั่วไปของคนแก่ธรรมดา ภาวะหลงลืมๆทั่วไปของคนแก่ที่สามารถพัฒนาเป็นโรคอัลไซเมอร์ในอนาคต หรือคนธรรมดา

0. สามารถเปิดดู dataset ได้ผ่านทาง preprocessing_test.py
1. เริ่มจากการนำ dataset ไป preprocess ด้วยโปรแกรม freesurfer ใน linux ด้วยโปรแกรม free_lnx.py
2. คัดสิ่งที่ต้องการ ซึ่งก็คือ brain.mgz จากผลลัพธ์ทั้งหมดของ freesurfer ผ่านทาง select_brain_nii.py
3. ทำการเปลี่ยนไฟล์ .mgz เป็น .nii ซึ่งพร้อมใช้เป็น dataset ด้วย mgz_to_nii2.py
4. ทำการแบ่ง train test และ augment ด้วย nii_preprocessing.py
5. train model ด้วย model_3_80_2.py
