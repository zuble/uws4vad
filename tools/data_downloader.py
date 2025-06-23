import os
import requests
import zlib
import hydra
import shutil
import urllib3
import zipfile
from zipfile import ZipFile
from pathlib import Path
from tqdm import tqdm
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BaseDatasetBuilder:
    def __init__(self, name, urls):
        self.name = name
        self.urls = urls

    def build(self, relocate_option, clean_option):
        print(f'Start downloading {self.name}')
        for i, (out_file, url) in tqdm(enumerate(self.urls.items()), bar_format="For {n}/{total} url"):
            print("Checking URL availability...")
            size = None
            if 'sharepoint' not in url:
                status, size = self._check_url(url)
                if not status:
                    print(f"URL is invalid.❌ Failed to download {self.name}")
                    return
                print("URL is valid. Starting download...")
            suffix = "({:d}/{:d})".format(i + 1, len(self.urls.items()))
            status = self._download_file(url, out_file, size, suffix)
            if status:
                print(f'Download complete for {out_file}.✅')
            else:
                print('Download failed.❌')
                return

        print(f'Start building dataset {self.name}')
        self._format(relocate_option)
        print('Building succeed.')
        self._clean_tmp_files(clean_option)

    def _check_url(self, url, timeout=10):
        """
        Check if the URL is valid.
        Returns (bool, int) indicating whether it's valid and its size in bytes.
        """
        try:
            resp = requests.head(
                url, allow_redirects=True, timeout=timeout, verify=False)
            if resp.status_code != 200:
                print(
                    f"Request failed with status code {resp.status_code} when checking {url}")
                return False, 0

            content_length = resp.headers.get('Content-Length', None)
            size = int(
                content_length) if content_length and content_length.isdigit() else 0
            return True, size

        except Exception as e:
            print(f"Request error: {e}")
            return False, 0

    def _download_file(self, url, output_path, total_size, suffix=None):
        """
        Download file from the URL, supporting http and sharepoint links.
        """
        if 'sharepoint' not in url:
            try:
                with requests.get(url, stream=True, verify=False) as r:
                    r.raise_for_status()
                    chunk_size = 1024 * 32
                    os.makedirs(Path(output_path).parent, exist_ok=True)
                    with open(output_path, 'wb') as f, tqdm(
                        total=total_size, unit='iB', unit_scale=True
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))
                return True
            except Exception as e:
                print(f"Download error: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False
        else:
            from DriveDownloader.downloader import download_single_file
            download_single_file(
                url, 
                filename=output_path, 
                thread_number=1,
                force_back_google=False, 
                list_suffix=suffix
            )
            return True

    def _format(self, relocate_option):
        raise NotImplementedError(
            f'{self.name} not implement method "_format".')

    def _clean_unzip_files(self):
        raise NotImplementedError(
            f'{self.name} not implement method "_clean_unzip_files".')

    def _clean_tmp_files(self, clean_option):
        if clean_option == 0:
            pass
        elif clean_option == 1:
            self._clean_unzip_files()
            print('Temporary files deleted; original files are saved.')
        elif clean_option == 2:
            shutil.rmtree(f'data/raw/{self.name}')
            print('All temporary files deleted.')
        else:
            raise RuntimeError(f'Invalid clean option {clean_option}')


class UCFCrimeBuilder(BaseDatasetBuilder):
    def __init__(self, urls):
        super().__init__(
            name = 'UCF-Crime',
            urls = urls
        )

    def _format(self, relocate_option):
        print('Start decompressing.')
        extract_zip_with_progress(
            zip_path = list(self.urls.keys())[0], 
            extract_to = 'data/raw/UCF-Crime'
        )
        print('Decompressing finished.')
        os.makedirs('data/UCF-Crime/Videos/Train', exist_ok=True)
        os.makedirs('data/UCF-Crime/Videos/Test', exist_ok=True)
        relocate_files_by_list(
            src_dir = 'data/raw/UCF-Crime/UCF_Crimes/Videos',
            dst_dir = 'data/UCF-Crime/Videos/Test',
            list_file='data/raw/UCF-Crime/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt',
            relocate_option=relocate_option
        )
        relocate_files_by_list(
            src_dir = 'data/raw/UCF-Crime/UCF_Crimes/Videos',
            dst_dir = 'data/UCF-Crime/Videos/Train',
            list_file = 'data/raw/UCF-Crime/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Train.txt',
            relocate_option = relocate_option
        )

    def _clean_unzip_files(self):
        shutil.rmtree('data/raw/UCF-Crime/UCF_Crimes')


class XDViolenceBuilder(BaseDatasetBuilder):
    def __init__(self, urls):
        super().__init__(name='XD-Violence', urls=urls)

    def _format(self, relocate_option):
        video_path = 'data/XD-Violence/Videos'
        os.makedirs(video_path, exist_ok=True)
        for key in tqdm(self.urls.keys(), bar_format="Formatting {n}/{total} raw file"):
            if 'Videos' in key:
                video_path_with_split = os.path.join(
                    video_path, 'Train') if 'Train' in key else os.path.join(video_path, 'Test')
                print('Start decompressing.')
                extract_zip_with_progress(
                    zip_path = key, 
                    extract_to = video_path_with_split
                )
                print('Decompressing finished.')

    def _clean_unzip_files(self):
        pass


def relocate_files_by_ext(src_dir, dst_dir, ext='.mp4', relocate_option='move'):
    if not os.path.isdir(src_dir):
        print(f"Source directory not existed: {src_dir}")
        return
    os.makedirs(dst_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(ext):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_dir, file)
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(dst_dir, f"{base}_{counter}{ext}")
                    counter += 1
                relocate(src_path, dst_path, relocate_option)
    print(f"Move from {src_dir} to {dst_dir}")


def extract_zip_with_progress(zip_path, extract_to):
    with ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            try:
                zip_ref.extract(member, extract_to)
            except (zipfile.BadZipFile, zlib.error) as e:
                print(f'{e}.\nBad zip file {member}.')


def relocate_files_by_list(src_dir, dst_dir, list_file, relocate_option='move'):
    os.makedirs(dst_dir, exist_ok=True)
    for file in list(open(list_file)):
        file = file.strip()
        if os.path.exists(os.path.join(src_dir, file)):
            relocate(os.path.join(src_dir, file), dst_dir, relocate_option)
        else:
            raise RuntimeError(f'File {file} does not exist.')


def relocate(src, dst, option):
    if option == 'move':
        shutil.move(src, dst)
    elif option == 'copy':
        shutil.copy(src, dst)
    else:
        raise ValueError(f'Relocate option {option} is invalid.')


@hydra.main(version_base=None, config_path="../src/config/tools", config_name="dataset_build")
def main(cfg):
    builder_dict = {'ucf': 'UCFCrimeBuilder', 'xd': 'XDViolenceBuilder'}
    for dataset in cfg.datasets:
        builder = eval(
            f'{builder_dict[dataset]}({getattr(cfg, f"{dataset}_urls")})')
        builder.build(cfg.relocate_option, cfg.clean_option)


if __name__ == "__main__":
    main()
