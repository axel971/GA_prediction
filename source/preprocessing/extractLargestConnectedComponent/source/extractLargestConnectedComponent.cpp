
#include <iostream>
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkImageMaskSpatialObject.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

using namespace std;
using namespace itk;


int main(int argc, char* argv[])
{

// Get the input mask
using pixelType = unsigned char;
using ImageType = itk::Image<pixelType, 3> ;
using ReaderTypeImage = itk::ImageFileReader<ImageType>;

ReaderTypeImage::Pointer readerImage = ReaderTypeImage::New();
readerImage->SetFileName(argv[1]);
readerImage->Update();


//Extract the largest connected component inside the mask image
using ConnectedComponentImageFilterType = itk::ConnectedComponentImageFilter<ImageType, ImageType>;
ConnectedComponentImageFilterType::Pointer connected = ConnectedComponentImageFilterType::New();
connected->SetInput(readerImage->GetOutput());
connected->SetBackgroundValue(0);
connected->Update();

using LabelShapeKeepNObjectsImageFilterType = itk::LabelShapeKeepNObjectsImageFilter<ImageType>;
LabelShapeKeepNObjectsImageFilterType::Pointer labelShapeKeepNObjectsImageFilter = LabelShapeKeepNObjectsImageFilterType::New();
labelShapeKeepNObjectsImageFilter->SetInput(connected->GetOutput());
labelShapeKeepNObjectsImageFilter->SetBackgroundValue(0);
labelShapeKeepNObjectsImageFilter->SetNumberOfObjects(1);
labelShapeKeepNObjectsImageFilter->SetAttribute(LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);

using RescaleFilterType = itk::RescaleIntensityImageFilter< ImageType, ImageType>;
RescaleFilterType::Pointer rescaleFilter = RescaleFilterType ::New();
rescaleFilter->SetOutputMinimum(0);
rescaleFilter->SetOutputMaximum(itk::NumericTraits<pixelType>::max());
rescaleFilter->SetInput(labelShapeKeepNObjectsImageFilter->GetOutput());

//Save the region of interest
using WriterTypeImage = itk::ImageFileWriter<ImageType>;
WriterTypeImage::Pointer writer = WriterTypeImage::New();
writer->SetFileName(argv[2]);
writer->SetInput(rescaleFilter->GetOutput());
writer->Update();

return 1;
}