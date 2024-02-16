# Development and software implementation of processing algorithms for a compact luminescence spectrograph

This repository contains the entire software development process for processing and improving the spectrum obtained by using a prototype of the _compact luminescence spectrograph_ (You can see the user manual in the file _Spectrograf_2022.pdf_)

### Philosophical meaning (purpose) of the project

Nowadays, the task of testing the quality of water from various sources and analyzing it for suitability for consumption is very important. However, currently used [methods](https://testslab.ru/stati/metody-analiza-vody/#:~:text=%D0%9B%D1%8E%D0%BC%D0%B8%D0%BD%D0%B5%D1%81%D1%86%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D0%B9%20%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D0%BF%D0%BE%D0%B7%D0%B2%D0%BE%D0%BB%D1%8F%D0%B5%D1%82%20%D0%BE%D0%B1%D0%BD%D0%B0%D1%80%D1%83%D0%B6%D0%B8%D1%82%D1%8C%20%D0%B8,%D1%81%D0%BE%D0%B5%D0%B4%D0%B8%D0%BD%D0%B5%D0%BD%D0%B8%D0%B9%20%D0%B2%20%D0%BF%D1%80%D0%BE%D0%B1%D0%B0%D1%85%20%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%8F%D1%8E%D1%82%20%D1%85%D1%80%D0%BE%D0%BC%D0%B0%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D1%8E) are often used only in laboratory conditions and are expensive and difficult to use. In turn, portable luminescent spectrographs, due to high requirements for precise alignment of the optical path and the quality of the dispersing element itself, [cost](https://aliexpress.ru/wholesale?SearchText=fluorescence+spectrophotometer&g=y&page=1&searchInfo=UeihUK8Yw0p7m0kK8rWUKQav9w%2F+9ysNUE3T21jdsWJLiHcifsoBkPSCd0kOhGpzFa0EOTe8gFMFJmEHO18K2KIYu6xu7G3Mqe%2FZXlsYbdJJOnwoRjWnsjJQNIjwp3qp4MeBFHghLjvAXQV2RMfCunpFB+N1qgjskc%2FKLQAv6DNdQraDNjY%3D) more than the average monthly salary.
So the idea of the project is to use a spectrograph assembled from cheap and low-precision materials and compensate for accuracy through software processing. The success of this work will greatly simplify the sample preparation and analysis of water, which can have a strong impact on many areas from monitoring sludge waters of hydraulic structures to simply determining the suitability of water for drinking in the field.

### Project goal
Thus, the task of this project is to compensate for such distortions occurring inside the optical path of the spectrograph as:
- Сhromatic aberration
- Astigmatism
- Parasitic illumination
- Noises
- And others


As well as improving image characteristics such as:
- Resolution
- Dynamic range
