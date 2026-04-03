import pandas as pd
import os
from pathlib import Path

def main():
    # 프로젝트 루트 경로 설정
    repo_root = Path(__file__).parent.parent
    base_dir = repo_root / "2026-ssafy-ai-15-2"
    
    train_path = base_dir / "train.csv"
    pred_path = base_dir / "submission.csv"
    output_path = base_dir / "train_incorrect_samples.csv"

    if not train_path.exists():
        print(f"❌ Error: {train_path} 파일이 없습니다.")
        return
    if not pred_path.exists():
        print(f"❌ Error: {pred_path} 파일이 없습니다. 추론을 완료하여 submission.csv를 생성해주세요.")
        return

    print("데이터 로딩 중...")
    train_df = pd.read_csv(train_path)
    pred_df = pd.read_csv(pred_path)

    # 1. id 기준으로 병합 (pred_df의 'answer' 컬럼을 'prediction'으로 변경)
    pred_df = pred_df.rename(columns={'answer': 'prediction'})
    merged_df = pd.merge(train_df, pred_df, on='id', how='inner')

    # 병합된 데이터가 없는 경우 (예: test 데이터를 추론한 경우) 예외 처리
    if len(merged_df) == 0:
        print("❌ Error: train.csv와 submission.csv 사이에 일치하는 id가 하나도 없습니다.")
        print(f"   - train.csv의 id 예시: {train_df['id'].iloc[0]}")
        print(f"   - submission.csv의 id 예시: {pred_df['id'].iloc[0]}")
        print("💡 현재 돌리고 계신 파이썬 추론 코드에서 입력 데이터를 'test.csv'가 아닌 'train.csv'로 변경하여 다시 실행해 주세요!")
        return

    # 2. 오답 데이터 필터링 (train.csv의 실제 정답 'answer'와 모델의 'prediction' 비교)
    incorrect_df = merged_df[merged_df['answer'] != merged_df['prediction']]

    # 3. 결과 저장
    incorrect_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 완료! 전체 {len(merged_df)}문제 중 모델이 틀린 문제 {len(incorrect_df)}개를 추출했습니다.")
    print(f"📁 오답 데이터 저장 위치: {output_path}")

if __name__ == "__main__":
    main()
